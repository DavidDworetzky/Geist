"""
Adapter that lets agents check the completion of asynchronous tool calls.

Async tool calls (adapter actions marked with @async_tool) return a job
handle immediately. This adapter is the agent-facing side of that contract:
the model polls check_async_tool with the handle's job_id until the job
reaches a terminal state, then reads the result.
"""
from typing import Any

from adapters.base_adapter import BaseAdapter


class JobStatusAdapter(BaseAdapter):

    def enumerate_actions(self):
        return ["check_async_tool"]

    def check_async_tool(self, job_id: int) -> dict[str, Any]:
        """Check whether an asynchronous tool call has finished.

        Returns the job's status (queued | running | succeeded | failed),
        its result when succeeded, and its error when failed. Poll again
        later if the status is not yet terminal.
        """
        # Imported here so adapter discovery does not require the app database.
        from app.models.database.job import get_job

        job = get_job(int(job_id))
        if job is None:
            return {"job_id": job_id, "status": "not_found"}
        state = job.to_dict()
        response: dict[str, Any] = {
            "job_id": state["job_id"],
            "status": state["status"],
            "done": state["status"] in ("succeeded", "failed"),
        }
        if state["status"] == "succeeded":
            response["result"] = state["result"]
        elif state["status"] == "failed":
            response["error"] = state["error"]
        return response
