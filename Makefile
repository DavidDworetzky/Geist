# Variables
PYTHON=uv run python
DOCKER_COMPOSE=docker compose

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
ifeq ($(MLX_BACKEND),1)
	$(DOCKER_COMPOSE) -f docker-compose.misc.yml up -d && $(PYTHON) bootstrap.py
else
	$(DOCKER_COMPOSE) up
endif

# Alias for run
.PHONY: up
up: run

# Run Python server only
.PHONY: server
server:
	$(PYTHON) bootstrap.py

# Run Python server with debugger (pdb)
.PHONY: debug
debug:
ifeq ($(MLX_BACKEND),1)
	$(DOCKER_COMPOSE) -f docker-compose.misc.yml up -d && uv run python -m pdb bootstrap.py
else
	$(DOCKER_COMPOSE) up -d && uv run python -m pdb bootstrap.py
endif

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
	$(PYTHON) initdb.py

# Build Docker images
.PHONY: build
build:
	$(DOCKER_COMPOSE) build 