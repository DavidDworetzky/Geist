import os

from app.services.job_queue import enqueue_or_reschedule, job_handler


MEMORY_JOB_KIND = "chat.memory.digest"
DEFAULT_MEMORY_IDLE_SECONDS = 20


def memory_idle_seconds() -> int:
    try:
        return max(0, int(os.getenv("GEIST_MEMORY_IDLE_SECONDS", "20")))
    except ValueError:
        return DEFAULT_MEMORY_IDLE_SECONDS


def schedule_chat_memory(
    user_id: int,
    chat_session_id: int,
    expected_revision: int,
    *,
    delay_seconds: int | None = None,
):
    delay = memory_idle_seconds() if delay_seconds is None else max(0, delay_seconds)
    return enqueue_or_reschedule(
        MEMORY_JOB_KIND,
        {
            "user_id": user_id,
            "chat_session_id": chat_session_id,
            "expected_revision": expected_revision,
            "pipeline_version": 1,
        },
        user_id=user_id,
        dedupe_key=f"chat-memory:{user_id}:{chat_session_id}",
        delay_seconds=delay,
    )


@job_handler(MEMORY_JOB_KIND)
def digest_chat_memory(payload: dict):
    from app.services.memory_processor import process_chat_memory

    return process_chat_memory(
        user_id=int(payload["user_id"]),
        chat_session_id=int(payload["chat_session_id"]),
        expected_revision=int(payload["expected_revision"]),
    )
