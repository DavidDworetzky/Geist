# Geist agent instructions

## Commands
- `docker compose up` to run the default Docker stack.
- `docker compose up -d` to run the default stack in the background for validation.
- `make build` to build Docker images.
- `make run MLX_BACKEND=1` to run Docker-managed services plus the native MLX backend.
- `make empty` to run an empty container for dependency work.
- `docker exec backend /bin/bash` to enter the backend container.

## Repository safety
- Do not read, print, grep, summarize, copy, or edit local `.env` files. Treat `.env`, `.env.*`, and other local secret files as off limits unless the user explicitly asks for a specific action. It is fine to inspect committed templates such as `.env.example`, `.env.sample`, or `.env.template`.
- Do not install Python, Node, Conda, Homebrew, or system packages without first prompting the user and getting explicit approval.
- When dependency installation is approved, prefer the existing project workflow. For Python, install inside the backend/container workflow and update the relevant environment lock/export file with `conda env export >> linux_environment.yml` after installing. For frontend packages, run installs under `client/geist`.
- Do not introduce new third-party dependencies when a minimal inline implementation or a Python/TypeScript standard library API is sufficient.

## Python style
- Write plain, idiomatic Python with small functions, explicit names, and straightforward control flow.
- Prefer project-local helpers and existing service/model patterns over new abstractions.
- Use type hints for new or changed function signatures when practical.
- Add docstrings only when they clarify non-obvious behavior or a public contract. Avoid boilerplate docstrings that restate the function name.
- Keep comments sparse and useful. Prefer readable code over explanatory comments.

## Package installs
- Backend package installs should happen inside the backend container unless the user approves a different workflow:
  - `docker exec backend /bin/bash`
  - `pip install PACKAGE`
  - `conda env export >> linux_environment.yml`
- Frontend package installs should happen from `client/geist`:
  - `cd client/geist && npm i PACKAGE`

## Running tests
- Default backend test command inside Docker:
  - `docker exec backend /bin/bash -lc "cd /opt/geist && PYTHONPATH=/opt/geist pytest"`
- Target backend tests when scope is narrow:
  - `docker exec backend /bin/bash -lc "cd /opt/geist && PYTHONPATH=/opt/geist pytest tests/agents/test_llama_mlx.py"`
- Frontend tests:
  - `cd client/geist && npm test -- --watchAll=false --passWithNoTests`
- Native MLX validation:
  - `make run MLX_BACKEND=1`

## Test expectations
- Test by default in Docker for backend changes.
- Test native MLX behavior when touching local inference, MLX runners, model loading, tokenization, generation, or code paths gated by `MLX_BACKEND=1`.
- If touching inference code under `agents/`, `agents/architectures/`, model runner registries, or completion logic, run focused agent/inference tests plus any affected service tests.
- If touching core contracts, run contract-focused tests. Core contracts include `agents/base_agent.py`, runner base classes/registries, Pydantic request/response models, API route schemas, database models that affect API behavior, and service interfaces consumed across modules.
- If a required Docker or native MLX test cannot be run in the current environment, state exactly what was not run and why.

## Agent hooks
- Repo-local Claude hooks live in `.claude/settings.json` and `.claude/hooks/`.
- Hooks block local `.env` reads and package installation commands unless the user explicitly approves them.
- Hooks also add a stop-time check that asks the agent to verify Docker and native MLX testing expectations for inference and core contract changes.

## SQLAlchemy
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

## Adding Models
Models should be added to scripts/copy_weights.py as well as the associated agent implementation inheriting from agents/base_agent. (GPT4 Agent, llama_agent, etc.)

## Preferences
prefer minimal inline implementations over extra dependency imports. Core libraries are better than pypi packages. 

## SDLC

## first, create a plan for your feature in /plans
## next, implement the plan, adding the backend data models, middle data models, service layer, routes, backend tests, 
## finally, test the solution by running `docker compose up -d`, then verifying no error logs in the docker container, then doing a curl command to localhost:3000
