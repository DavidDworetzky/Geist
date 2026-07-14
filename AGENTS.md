#commands
`docker compose up` to run the docker container
`make build` to build the solution
`make run MLX_BACKEND=1` to run the solution with MLX_BACKEND instead of
`make empty` to run an empty container to install dependencies
#package installs
`uv add PACKAGE==VERSION` to add pinned dependencies and update `pyproject.toml` and `uv.lock` together.
The Docker image installs from the same lockfile, so no separate environment export is needed.
Run a Python dependency audit before committing dependency changes.
#frontend package installs
`cd client/geist && npm install --package-lock-only --ignore-scripts --save-exact PACKAGE@VERSION`
Use `npm ci --ignore-scripts --audit=false --fund=false` for frontend installs from the committed lockfile.
Run `npm audit --package-lock-only` before committing frontend dependency changes.
#running tests
`cd /opt/geist && PYTHONPATH=/opt/geist pytest` in the backend container

#pre-push AI testing
Before pushing, publishing, or opening a PR, use `.agents/skills/geist-test-loop/SKILL.md` when feasible. The loop should collect evidence from relevant fast checks, Docker startup/logs/curl, browser UI smoke testing, basic chat, settings defaults, and native `make run MLX_BACKEND=1` when the change touches native/local model behavior.

#pre-commit hooks
Use `pre-commit install` to enable local hooks. Keep hooks fast: linting, formatting, type checks, staged secret scanning, and basic frontend lint are appropriate here. Full Docker/native/browser smoke testing belongs in the AI pre-push test loop, not in pre-commit.

#repository safety
Do not read, print, grep, summarize, copy, or edit local `.env` files. Treat `.env`, `.env.*`, and other local secret files as off limits unless the user explicitly asks for a specific action. It is fine to inspect committed templates such as `.env.example`, `.env.sample`, or `.env.template`.
Do not install Python, Node, Conda, Homebrew, or system packages without first prompting the user and getting explicit approval.
Do not introduce new third-party dependencies when a minimal inline implementation or a Python/TypeScript standard library API is sufficient.

#python style
Write plain, idiomatic Python with small functions, explicit names, and straightforward control flow.
Prefer project-local helpers and existing service/model patterns over new abstractions.
Use type hints for new or changed function signatures when practical.
Add docstrings only when they clarify non-obvious behavior or a public contract. Avoid boilerplate docstrings that restate the function name.
Keep comments sparse and useful. Prefer readable code over explanatory comments.

#test expectations
Test by default in Docker for backend changes.
Test native MLX behavior when touching local inference, MLX runners, model loading, tokenization, generation, or code paths gated by `MLX_BACKEND=1`.
If touching inference code under `agents/`, `agents/architectures/`, model runner registries, or completion logic, run focused agent/inference tests plus any affected service tests.
If touching core contracts, run contract-focused tests. Core contracts include `agents/base_agent.py`, runner base classes/registries, Pydantic request/response models, API route schemas, database models that affect API behavior, and service interfaces consumed across modules.
If a required Docker or native MLX test cannot be run in the current environment, state exactly what was not run and why.

#agent hooks
Repo-local Claude hooks live in `.claude/settings.json` and `.claude/hooks/`.
Hooks block local `.env` reads and package installation commands unless the user explicitly approves them.
Hooks also add a stop-time check that asks the agent to verify Docker and native MLX testing expectations for inference and core contract changes.

#SQLAlchemy
When adding classes to sql alchemy, take the following example
EXAMPLE:
import datetime
from sqlalchemy import Column, Integer, String, ForeignKey, LargeBinary, DateTime, Boolean, ARRAY, DateTime
from sqlalchemy.orm import relationship, Session
from app.models.database.database import Base, Session
from sqlalchemy.dialects.postgresql import insert

class AgentPreset(Base):
    """
    Class used to represent a preset for an agent.
    """
    __tablename__ = 'agent_preset'
    agent_preset_id = Column(Integer, primary_key=True, autoincrement=True)
    #language model presets
    name = Column(String)
    version = Column(String)
    description = Column(String)
    max_tokens = Column(Integer)
    n = Column(Integer)
    temperature = Column(Integer)
    top_p = Column(Integer)
    frequency_penalty = Column(Integer)
    presence_penalty = Column(Integer)
    tags = Column(String)
    #memory presets
    working_context_length = Column(Integer)
    long_term_context_length = Column(Integer)
    agent_type = Column(String)
    #prompt presets
    prompt = Column(String)
    #interactive_only - is not an independent agent.
    interactive_only = Column(Boolean)
    # optional processing settings
    process_world = Column(Boolean)
    #restriction relationships
    restrictions = relationship("Restriction", back_populates="agent_preset")
    create_date = Column(DateTime)
    update_date = Column(DateTime)

classes are stored in app >> models >> database.

#Adding Models
Models should be added to scripts/copy_weights.py as well as the associated agent implementation inheriting from agents/base_agent. (GPT4 Agent, llama_agent, etc.)

#preferences
prefer minimal inline implementations over extra dependency imports. Core libraries are better than pypi packages.

#SDLC

## first, create a plan for your feature in /plans
## next, implement the plan, adding the backend data models, middle data models, service layer, routes, backend tests,
## finally, test the solution by running `docker compose up -d`, then verifying no error logs in the docker container, then doing a curl command to localhost:3000
