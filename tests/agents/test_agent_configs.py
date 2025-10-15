from agents.agent_registry import get_all_base_agent_classes
VALID_AGENTS = ['GPT4Agent', 'LlamaAgent', 'OnlineAgent', 'LocalAgent']

def test_assert_valid_agents():
    agent_classes = get_all_base_agent_classes()
    for agent in agent_classes:
        assert agent.__name__ in VALID_AGENTS