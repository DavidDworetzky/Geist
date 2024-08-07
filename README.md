# Geist
FOSS Project for natural language AI Assistants - using GPT, LlaMA, Mistral etc. architectures.
Geist is a framework for multiple natural language AI Assistants to interact, train, and do work.

# Roadmap
1. Create core OODA architecture for AI Assistants using various LLM transformer architectures.
2. Create core API surface area for inference for architectures.
3. Persist conversations, memory, and presets for various agents and feed these back into agent startup / initialization.
4. Create external world adapters for agents to interact with the real world, such as VoIP, Slack, Notion, etc.

# Roadmap Agents
1. Software engineer spirit.  (Reviews PRs, creates design docs, creates code from natural language description of features.)
2. Virtual assistant spirit.  (Summarize and respond to emails, keep track of birthdays, 
3. Business manager spirit.   (Generate business ideas, affiliate marketing, etc. and provide detailed blueprints for execution.)
4. Research assistant spirit. (Summarize arxiv posts on machine learning / deep learning, provide briefs on latest changes in the field.)

# Core Architecture

The Geist Architecture consists of three main components: a world model, a task creation model, and an execution-based  model. The diagram below illustrates the architecture and relationships between these components.

```mermaid
flowchart LR
    A[World Model] -->|Updates| B[Task Creation Model]
    B -->|Generates Tasks| C[Execution-based LLM Model]
    C -->|Execution Results| A
    D[Plugin System] -->|Provides Capabilities| C
    E[Long-term Memory] -->|Base Knowledge| A
    F[Short-term Memory] -->|Context-specific Info| C
```

# Versioning and Setup
## Install Postgresql
1. Version 16.2 on Mac or Windows

## Install Miniconda
1. Create a python 3.10 environment solve and install with windows_x64_environment.yml or mac_arm_environment.yml

## Starting the solution
1. Start the postgresql server `PATH/pg_ctl -D DATA_PATH -l LOG_PATH start` 
2. Make sure that your .env file is initialized with the following values:
    - OPENAI_TOKEN = TOKEN
    - POSTGRES_PWD = PASSWORD
    - POSTGRES_DB = geist
    - TWILIO_SID = ACCOUNT_SIDE
    - TWILIO_SOURCE = SOURCE_NUMBER
    - TWILIO_TOKEN = API_TOKEN
    - ENHANCED_LOGGING = FALSE







