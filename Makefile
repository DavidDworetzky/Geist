# Variables
UV=uv
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

# Default target
.PHONY: all
all: help

# Help target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  make run         - Run the backend natively with uv (SQLite by default, no Docker)"
	@echo "  make server      - Alias for run"
	@echo "  make debug       - Run the backend natively with debugger (pdb)"
	@echo "  make sync        - Install/refresh the uv-managed environment from uv.lock"
	@echo "  make init-db     - Initialize the database (SQLite by default)"
	@echo "  make run-docker  - Run the full Docker stack (backend + PostgreSQL + frontend)"
	@echo "  make services    - Run auxiliary Docker services only (PostgreSQL + frontend)"
	@echo "  make docker      - Alias for run-docker (detached)"
	@echo "  make stop        - Stop all running services"
	@echo "  make clean       - Clean up Docker resources"
	@echo "  make build       - Build Docker images"

# Install/refresh the local environment from the lockfile
.PHONY: sync
sync:
	$(UV) sync

# Run the backend natively. Uses SQLite unless GEIST_DATABASE_PROVIDER says otherwise.
.PHONY: run
run:
ifeq ($(IS_WSL),1)
ifeq ($(MLX_BACKEND),1)
	@echo "WSL detected; MLX_BACKEND=1 uses the Docker stack via Windows Docker Desktop."
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) up
else
	$(UV) run python initdb.py
	PYTHONUNBUFFERED=1 $(UV) run python bootstrap.py
endif
else
	$(UV) run python initdb.py
	PYTHONUNBUFFERED=1 $(UV) run python bootstrap.py
endif

# Alias for run
.PHONY: up server
up: run
server: run

# Run the backend natively with debugger (pdb)
.PHONY: debug
debug:
	$(UV) run python initdb.py
	PYTHONUNBUFFERED=1 $(UV) run python -m pdb bootstrap.py

# Initialize database (SQLite by default; set GEIST_DATABASE_PROVIDER=postgresql for Postgres)
.PHONY: init-db
init-db:
	$(UV) run python initdb.py

# Run the full Docker stack (backend + PostgreSQL + frontend)
.PHONY: run-docker
run-docker:
ifeq ($(IS_WSL),1)
	@echo "WSL detected; using Windows Docker Desktop via docker.exe compose."
endif
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) up

# Run auxiliary Docker services only (PostgreSQL + frontend), e.g. alongside a native backend
.PHONY: services
services:
	$(DOCKER_COMPOSE) -f docker-compose.misc.yml up -d

# Run the full Docker stack detached
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

# Build Docker images
.PHONY: build
build:
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) build