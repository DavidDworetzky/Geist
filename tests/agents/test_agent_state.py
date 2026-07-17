from agents.models.agent_state import ConversationState
from agents.models.chat_result import ToolCallResult
from agents.models.tool_calling import ChatMessage


def test_conversation_state_keeps_agent_runs_scoped_to_one_chat():
    conversation = ConversationState(chat_id=12, user_id=7)
    conversation.add_system_prompt("Be concise.")
    conversation.hydrate(
        [
            ChatMessage(role="user", content="Remember cobalt."),
            ChatMessage(role="assistant", content="I will remember cobalt."),
        ]
    )

    run = conversation.begin_run("What should you remember?")

    assert [message.role for message in run.model_messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert run.model_messages[-1].content == "What should you remember?"
    assert conversation.messages[-1].content == "I will remember cobalt."


def test_agent_run_context_snapshots_terminal_execution_state():
    conversation = ConversationState(chat_id=None, user_id=3)
    run = conversation.begin_run("Find the file")
    run.record_assistant(
        ChatMessage(
            role="assistant",
            content="I found it.",
        )
    )
    run.record_tool_call(
        ToolCallResult.create(
            id="call_1",
            name="documents.search",
            status="succeeded",
            result_summary="Found the file",
        )
    )
    run.transition("completed")

    snapshot = run.persistence_snapshot()

    assert snapshot["session_id"] is None
    assert snapshot["run_id"] == run.run_id
    assert snapshot["status"] == "completed"
    assert snapshot["new_ai_message"] == "I found it."
    assert snapshot["transcript"] == [
        {"role": "user", "content": "Find the file"},
        {"role": "assistant", "content": "I found it."},
    ]
    assert snapshot["tool_calls"][0].id == "call_1"

    run.mark_persisted(41)
    assert conversation.chat_id == 41
    assert run.persisted_status == "completed"
