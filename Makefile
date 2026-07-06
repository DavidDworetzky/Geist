# Variables
PYTHON=python
IS_WSL=$(shell grep -qi microsoft /proc/version 2>/dev/null && echo 1 || echo 0)

ifeq ($(IS_WSL),1)
DOCKER=docker.exe
DOCKER_COMPOSE=docker.exe compose
COMPOSE_FILE=docker-compose.wsl.yml
else
DOCKER=docker
DOCKER_COMPOSE=docker compose
COMPOSE_FILE=docker-compose.yml
endif

CONDA_ENV=geist-mac
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
	@echo "  make debug       - Run Python bootstrap server with debugger (pdb)"
	@echo "  make docker      - Run Docker services only"
	@echo "  make stop        - Stop all running services"
	@echo "  make clean       - Clean up Docker resources"
	@echo "  make init-db     - Initialize the database"
	@echo "  make build       - Build Docker images"

# Run both server and Docker services
.PHONY: run
run:
ifeq ($(IS_WSL),1)
	@echo "WSL detected; using Windows Docker Desktop via docker.exe compose."
ifeq ($(MLX_BACKEND),1)
	@echo "MLX_BACKEND=1 uses a native Unix/Conda backend and is not supported on Windows/WSL."
	@echo "Starting the full Docker stack instead."
endif
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) up
else
ifeq ($(MLX_BACKEND),1)
	$(DOCKER_COMPOSE) -f docker-compose.misc.yml up -d && $(CONDA_ACTIVATE) && PYTHONUNBUFFERED=1 $(PYTHON) bootstrap.py
else
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) up
endif
endif

# Alias for run
.PHONY: up
up: run

# Run Python server only
.PHONY: server
server:
	$(CONDA_ACTIVATE) && $(PYTHON) bootstrap.py

# Run Python server with debugger (pdb)
.PHONY: debug
debug:
ifeq ($(MLX_BACKEND),1)
	$(DOCKER_COMPOSE) -f docker-compose.misc.yml up -d && $(CONDA_ACTIVATE) && PYTHONUNBUFFERED=1 $(PYTHON) -m pdb bootstrap.py
else
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) up -d && $(CONDA_ACTIVATE) && PYTHONUNBUFFERED=1 $(PYTHON) -m pdb bootstrap.py
endif

# Run Docker services only
.PHONY: docker
docker:
ifeq ($(IS_WSL),1)
	@echo "WSL detected; using Windows Docker Desktop via docker.exe compose."
endif
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) up -d

# Stop all services
.PHONY: stop
stop:
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) down
ifeq ($(IS_WSL),1)
	@echo "WSL detected; skipping Unix pkill cleanup."
else
	@echo "Checking for running Python processes..."
	-pkill -f "python bootstrap.py"
endif

# Clean up Docker resources
.PHONY: clean
clean:
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) down -v --remove-orphans
	$(DOCKER) system prune -f

# Initialize database
.PHONY: init-db
init-db:
	$(CONDA_ACTIVATE) && $(PYTHON) initdb.py

# Build Docker images
.PHONY: build
build:
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) build