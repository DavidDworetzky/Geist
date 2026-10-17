# Background file processing and streaming UI

Reuse the durable single-worker job queue for upload extraction and reprocessing.
Upload storage runs in the request thread pool; extraction runs after enqueue.
Return a workspace-owned job ID and let upload UI observe its terminal status.
Workers recheck file ownership and store only a compact result. Failed extraction
marks the job failed; explicit reprocessing can retry. No database schema changes.
Synchronous file endpoints run in FastAPI's thread pool.

Batch text deltas per animation frame, flushing before control/terminal events
and cancellation. Discard pending updates on reset, replacement, or unmount.
Memoize individual chat turns so completed turns do not render on every delta.

Validate API enqueue/ownership/failure contracts, worker execution, stream ordering,
cancellation/reset cleanup, memoization, and upload progress. Run focused Python
and React suites, lint/type checks, and feasible runtime smoke checks.
