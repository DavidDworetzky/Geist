# Variables
UV=uv
DOCKER_COMPOSE=docker compose

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
	@echo "  make services    - Run the Docker frontend alongside the native SQLite backend"
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
	$(UV) run python initdb.py
	PYTHONUNBUFFERED=1 $(UV) run python bootstrap.py

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
	$(DOCKER_COMPOSE) up

# Run the Docker frontend alongside the native SQLite backend
.PHONY: services
services:
	$(DOCKER_COMPOSE) -f docker-compose.misc.yml up -d

# Run the full Docker stack detached
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

# Build Docker images
.PHONY: build
build:
	$(DOCKER_COMPOSE) build
