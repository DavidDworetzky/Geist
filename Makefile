# Variables
PYTHON=python
DOCKER_COMPOSE=docker compose
CONDA_ENV=mac_environment
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh && conda activate $(CONDA_ENV)

# Default target
.PHONY: all
all: help

# Help target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  make run         - Run both backend server and Docker services"
	@echo "  make server      - Run Python bootstrap server only"
	@echo "  make docker      - Run Docker services only"
	@echo "  make stop        - Stop all running services"
	@echo "  make clean       - Clean up Docker resources"
	@echo "  make init-db     - Initialize the database"

# Run both server and Docker services
.PHONY: run
run:
	$(DOCKER_COMPOSE) up -d
	$(CONDA_ACTIVATE) && $(PYTHON) bootstrap.py

# Run Python server only
.PHONY: server
server:
	$(CONDA_ACTIVATE) && $(PYTHON) bootstrap.py

# Run Docker services only
.PHONY: docker
docker:
	$(DOCKER_COMPOSE) up -d

# Stop all services
.PHONY: stop
stop:
	$(DOCKER_COMPOSE) down
	@echo "Checking for running Python processes..."
	-pkill -f "python bootstrap.py"

# Clean up Docker resources
.PHONY: clean
clean:
	$(DOCKER_COMPOSE) down -v --remove-orphans
	docker system prune -f

# Initialize database
.PHONY: init-db
init-db:
	$(CONDA_ACTIVATE) && $(PYTHON) initdb.py 