#commands
`docker compose up` to run the docker container
`make build` to build the solution
`make run MLX_BACKEND=1` to run the solution with MLX_BACKEND instead of
`make empty` to run an empty container to install dependencies

#script parameter conventions
All scripts in the `scripts/` directory should follow these command-line parameter conventions:

**Required Conventions:**
- Use `--` prefix (double dash) for all named parameters
- Use `argparse` for argument parsing
- Provide clear `--help` documentation for all parameters
- Include type hints and default values where appropriate

**Acceptable Patterns:**
- Short aliases (e.g., `-v` for `--verbose`, `-c` for `--config`) are acceptable
- Boolean flags using `action="store_true"` (e.g., `--dry-run`, `--commit`)
- Subcommand architectures using `subparsers` for complex scripts

**Examples:**
```bash
# Good: Named parameters with -- prefix
python scripts/download_models.py --model_id meta-llama/Meta-Llama-3.1-8B-Instruct --weights_dir app/model_weights

# Good: Short aliases
python scripts/sync_models.py -v --provider openai

# Good: Boolean flags
python scripts/insert_presets.py --commit --overwrite

# Avoid: Positional arguments without names
python scripts/download_models.py meta-llama/Meta-Llama-3.1-8B-Instruct  # Bad
```

When creating new scripts, always use `argparse.ArgumentParser()` and define parameters with descriptive help text.

#package installs
`docker exec backend /bin/bash` to enter the backend container
`pip install PACKAGE` to install dependencies
When installing packages, `conda env export >> linux_environment.yml` after installing to freeze installs. 
#frontend package installs
`cd client & npm i PACKAGE`
#running tests
`cd /opt/geist && PYTHONPATH=/opt/geist pytest` in the backend container


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


#Plan Files
Good plan file formats should look like /plans/144-SETTINGS-UI.md. They should be concise, should feature what of an implementation already exists, and what needs to be added. They should feature component level descriptions of what should be implemented in a solution. 