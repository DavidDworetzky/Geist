from enum import Enum

class AgentType(Enum):
    LLAMA = "LLAMA"
    GPT4AGENT = "GPT4AGENT"
    #using completions api
    HTTPAGENT = "HTTPAGENT"
    #using local agent through MLX or CUDA architectures
    LOCALAGENT = "LOCALAGENT"
