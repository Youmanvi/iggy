// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::sync::Arc;

use dlopen2::wrapper::Container;
use tokio::runtime::Handle;
use tracing::warn;

use crate::metrics::ConnectorType;
use crate::{SinkApi, SourceApi, close_plugin_instance};

/// A plugin's `iggy_{source,sink}_close` together with whatever keeps the
/// library that exports it mapped, so the call stays valid once it is deferred
/// off the calling thread.
pub(crate) type PluginClose = Arc<dyn Fn(u32) -> i32 + Send + Sync>;

/// Closes a plugin instance that `iggy_{source,sink}_open` created and nothing
/// else will ever reach.
///
/// Between the open succeeding and the plugin id reaching the connector's
/// details, the instance exists inside the plugin and nothing outside it knows
/// the id: `stop_connector` closes whatever `details.info.id` holds, which is
/// still the previous instance. An early return there stranded the new one for
/// the life of the process. A guard rather than a cleanup branch per fallible
/// call, because the window is those two statements rather than whichever call
/// between them is fallible today, so a `?` added inside it stays correct.
///
/// Teardown runs two ways and they are not interchangeable. [`Self::close`]
/// awaits, so an error returned after it means the instance is already gone
/// and an immediate retry has nothing to collide with. `Drop` cannot await, so
/// it hands the work to the blocking pool; the closure carries the container,
/// which is what keeps the library mapped until the call returns.
#[must_use = "dropping an armed guard closes the plugin instance"]
pub(crate) struct PluginInstanceGuard {
    /// `Some` while this guard owns the instance, `None` once something else
    /// does. One representation rather than a close plus a flag that had to
    /// agree with it, and taking it is what lets both teardown paths run
    /// without cloning the callback.
    close: Option<PluginClose>,
    kind: ConnectorType,
    plugin_id: u32,
    key: String,
}

impl PluginInstanceGuard {
    /// Arms a guard over a source instance the caller has just opened through
    /// `container`, which it captures rather than borrows for the reason the
    /// type documents.
    pub(crate) fn for_source(
        container: Arc<Container<SourceApi>>,
        plugin_id: u32,
        key: &str,
    ) -> Self {
        Self::new(
            Arc::new(move |id| (container.iggy_source_close)(id)),
            ConnectorType::Source,
            plugin_id,
            key,
        )
    }

    /// The sink counterpart of [`Self::for_source`].
    pub(crate) fn for_sink(container: Arc<Container<SinkApi>>, plugin_id: u32, key: &str) -> Self {
        Self::new(
            Arc::new(move |id| (container.iggy_sink_close)(id)),
            ConnectorType::Sink,
            plugin_id,
            key,
        )
    }

    /// Kept behind the typed constructors so no production caller can build a
    /// guard that holds a close pointer without its library. Tests pass a
    /// closure.
    fn new(close: PluginClose, kind: ConnectorType, plugin_id: u32, key: &str) -> Self {
        Self {
            close: Some(close),
            kind,
            plugin_id,
            key: key.to_owned(),
        }
    }

    /// Hands ownership of the instance to the caller, once something else can
    /// close it. Call only after the plugin id is recorded on the connector's
    /// details.
    pub(crate) fn disarm(mut self) {
        self.close = None;
    }

    /// The awaited half of the teardown the type documents. Error arms call it
    /// rather than leaving the work to `Drop`, which cannot offer the ordering.
    pub(crate) async fn close(mut self) {
        let Some(close) = self.close.take() else {
            return;
        };
        let kind = self.kind;
        let plugin_id = self.plugin_id;
        let key = std::mem::take(&mut self.key);
        if tokio::task::spawn_blocking(move || {
            close_plugin_instance(close.as_ref(), kind, plugin_id, &key)
        })
        .await
        .is_err()
        {
            let kind = kind.as_label();
            warn!(
                "Teardown of failed {kind} connector with ID: {plugin_id} did not run to completion."
            );
        }
    }
}

impl Drop for PluginInstanceGuard {
    fn drop(&mut self) {
        let Some(close) = self.close.take() else {
            return;
        };

        let kind = self.kind;
        let plugin_id = self.plugin_id;
        let key = std::mem::take(&mut self.key);
        // The SDK containers drive the plugin's own `close()` under `block_on`
        // and run for as long as the plugin takes, so it goes to the blocking
        // pool where blocking is what the thread is for.
        match Handle::try_current() {
            Ok(handle) => {
                handle.spawn_blocking(move || {
                    close_plugin_instance(close.as_ref(), kind, plugin_id, &key)
                });
            }
            // No runtime to hand it to, and no worker to protect either.
            Err(_) => close_plugin_instance(close.as_ref(), kind, plugin_id, &key),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::RuntimeError;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::time::Duration;

    static TEST_PLUGIN_ID: AtomicU32 = AtomicU32::new(u32::MAX / 2);

    fn next_plugin_id() -> u32 {
        TEST_PLUGIN_ID.fetch_add(1, Ordering::Relaxed)
    }

    /// A close that records the ids it was handed and answers `result`.
    ///
    /// The guard takes its close as a closure, so each test owns its recorder
    /// and nothing is shared between tests. An `extern "C" fn` cannot capture,
    /// which is what used to force this through statics.
    fn recording_close(result: i32) -> (PluginClose, Arc<Mutex<Vec<u32>>>) {
        let closed = Arc::new(Mutex::new(Vec::new()));
        let recorded = closed.clone();
        (
            Arc::new(move |id| {
                recorded.lock().expect("close recorder").push(id);
                result
            }),
            closed,
        )
    }

    /// The shape `start_connector` has: a guard armed over an instance nothing
    /// else knows about, then a fallible step whose `?` returns before anything
    /// records the id.
    fn start_with_fallible_step(
        close: PluginClose,
        plugin_id: u32,
        step: Result<(), RuntimeError>,
    ) -> Result<(), RuntimeError> {
        let instance_guard =
            PluginInstanceGuard::new(close, ConnectorType::Source, plugin_id, "random");
        step?;
        instance_guard.disarm();
        Ok(())
    }

    #[test]
    fn given_armed_guard_when_dropped_should_close_the_instance() {
        // The leak this exists for: the open has created the instance and
        // nothing outside the plugin knows its id yet, so an early return here
        // would strand it for the life of the process.
        let plugin_id = next_plugin_id();
        let (close, closed) = recording_close(0);

        drop(PluginInstanceGuard::new(
            close,
            ConnectorType::Source,
            plugin_id,
            "random",
        ));

        assert_eq!(
            *closed.lock().expect("close recorder"),
            vec![plugin_id],
            "a guard still armed owns the instance and must close exactly it"
        );
    }

    #[test]
    fn given_fallible_step_when_it_returns_early_should_close_the_instance() {
        // The shape dropping or disarming inline cannot show, and the one the
        // guard is there for: the `?` leaves with the guard still armed and
        // never reaches `disarm`. The error arms call `close()` directly now,
        // so this is the net under a `?` added inside the window later.
        let plugin_id = next_plugin_id();
        let (close, closed) = recording_close(0);

        let result = start_with_fallible_step(
            close,
            plugin_id,
            Err(RuntimeError::InvalidConfiguration("injected".to_string())),
        );

        assert!(result.is_err(), "the injected failure has to propagate");
        assert_eq!(
            *closed.lock().expect("close recorder"),
            vec![plugin_id],
            "a `?` must not strand the instance it left behind"
        );
    }

    #[test]
    fn given_fallible_step_when_it_succeeds_should_leave_the_instance_open() {
        // The other half of the same helper: reaching `disarm` hands the
        // instance on rather than closing it.
        let plugin_id = next_plugin_id();
        let (close, closed) = recording_close(0);

        let result = start_with_fallible_step(close, plugin_id, Ok(()));

        assert!(result.is_ok());
        assert!(
            closed.lock().expect("close recorder").is_empty(),
            "a step that succeeded leaves the instance for the manager to close"
        );
    }

    #[tokio::test]
    async fn given_armed_guard_when_dropped_in_runtime_should_close_off_the_worker() {
        // `drop` cannot await, and the SDK container's close drives the plugin's
        // own teardown under `block_on`, so closing here would hold a worker for
        // however long the plugin takes. It goes to the blocking pool instead,
        // which is what the differing thread asserts. The close still has to
        // happen.
        let plugin_id = next_plugin_id();
        let dropping_thread = std::thread::current().id();
        let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();

        drop(PluginInstanceGuard::new(
            Arc::new(move |id| {
                let _ = sender.send((id, std::thread::current().id()));
                0
            }),
            ConnectorType::Source,
            plugin_id,
            "random",
        ));

        let (closed_id, closing_thread) =
            tokio::time::timeout(Duration::from_secs(5), receiver.recv())
                .await
                .expect("the deferred close should run")
                .expect("the deferred close should report the instance");
        assert_eq!(
            closed_id, plugin_id,
            "the deferred close must reach the instance the guard was armed over"
        );
        assert_ne!(
            closing_thread, dropping_thread,
            "closing on the dropping thread holds it for the plugin's teardown"
        );
    }

    #[tokio::test]
    async fn given_armed_guard_when_closed_should_finish_before_returning() {
        // What the error arms rely on: once `close()` has returned the instance
        // is gone, so the error they return cannot reach an operator who then
        // retries into a collision with it.
        let plugin_id = next_plugin_id();
        let (close, closed) = recording_close(0);

        PluginInstanceGuard::new(close, ConnectorType::Source, plugin_id, "random")
            .close()
            .await;

        assert_eq!(
            *closed.lock().expect("close recorder"),
            vec![plugin_id],
            "close() has to await the teardown, and the drop after it must not repeat it"
        );
    }

    #[test]
    fn given_refused_close_when_guard_drops_should_close_once_and_swallow_refusal() {
        // The plugin answers -1 for an id it does not know. Both callers are
        // already returning an error of their own, so the refusal is reported
        // and not propagated: unwinding out of `drop` would be worse than the
        // leak it is cleaning up after. The harness gives no-panic for free, so
        // what this asserts is the single call.
        let plugin_id = next_plugin_id();
        let (close, closed) = recording_close(-1);

        drop(PluginInstanceGuard::new(
            close,
            ConnectorType::Source,
            plugin_id,
            "random",
        ));

        assert_eq!(
            *closed.lock().expect("close recorder"),
            vec![plugin_id],
            "a refusal must not become a retry or a second close"
        );
    }
}
