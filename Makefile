# Variables
PYTHON=python
DOCKER_COMPOSE=docker compose
CONDA_ENV=geist-mac
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh && conda activate $(CONDA_ENV)

# Default target
.PHONY: all
all: help

# Help target
.PHONY: help
help:
	@echo "Available targets:"
	@echo ""
	@echo "Python (Legacy):"
	@echo "  make run         - Run both backend server and Docker services"
	@echo "  make server      - Run Python bootstrap server only"
	@echo "  make debug       - Run Python bootstrap server with debugger (pdb)"
	@echo "  make docker      - Run Docker services only"
	@echo "  make stop        - Stop all running services"
	@echo "  make clean       - Clean up Docker resources"
	@echo "  make init-db     - Initialize the database"
	@echo "  make build       - Build Docker images"
	@echo ""
	@echo "Go Server:"
	@echo "  make go-build        - Build Go server"
	@echo "  make go-test         - Run Go tests"
	@echo "  make go-run          - Run Go server"
	@echo "  make go-lint         - Lint Go code"
	@echo ""
	@echo "Bazel:"
	@echo "  make bazel-build     - Build with Bazel"
	@echo "  make bazel-test      - Test with Bazel"
	@echo ""
	@echo "MLX Service:"
	@echo "  make mlx-run         - Run MLX inference service"
	@echo ""
	@echo "Docker (Go):"
	@echo "  make go-docker-build - Build Go Docker images"
	@echo "  make go-docker-up    - Start Go services"
	@echo "  make go-docker-down  - Stop Go services"

# Run both server and Docker services
.PHONY: run
run:
ifeq ($(MLX_BACKEND),1)
	$(DOCKER_COMPOSE) -f docker-compose.misc.yml up -d && $(CONDA_ACTIVATE) && PYTHONUNBUFFERED=1 $(PYTHON) bootstrap.py
else
	$(DOCKER_COMPOSE) up
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
	$(DOCKER_COMPOSE) up -d && $(CONDA_ACTIVATE) && PYTHONUNBUFFERED=1 $(PYTHON) -m pdb bootstrap.py
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
	$(CONDA_ACTIVATE) && $(PYTHON) initdb.py

# Build Docker images
.PHONY: build
build:
	$(DOCKER_COMPOSE) build

# ============================================================================
# Go Server Targets
# ============================================================================

# Build the Go server
.PHONY: go-build
go-build:
	@echo "Building Go server..."
	cd go && go build -o ../bin/geist-server ./cmd/server

# Run Go tests
.PHONY: go-test
go-test:
	@echo "Running Go tests..."
	cd go && go test -v ./...

# Run Go tests with coverage
.PHONY: go-test-coverage
go-test-coverage:
	@echo "Running Go tests with coverage..."
	cd go && go test -coverprofile=coverage.out ./...
	cd go && go tool cover -html=coverage.out -o coverage.html

# Run the Go server
.PHONY: go-run
go-run:
	@echo "Starting Go server..."
	cd go && go run ./cmd/server

# Lint Go code
.PHONY: go-lint
go-lint:
	@echo "Linting Go code..."
	cd go && golangci-lint run ./...

# Format Go code
.PHONY: go-fmt
go-fmt:
	@echo "Formatting Go code..."
	cd go && go fmt ./...

# Download Go modules
.PHONY: go-mod
go-mod:
	@echo "Downloading Go modules..."
	cd go && go mod download && go mod tidy

# ============================================================================
# Bazel Targets
# ============================================================================

# Build with Bazel
.PHONY: bazel-build
bazel-build:
	@echo "Building with Bazel..."
	bazel build //go/cmd/server:server

# Test with Bazel
.PHONY: bazel-test
bazel-test:
	@echo "Testing with Bazel..."
	bazel test //go/...

# Clean Bazel cache
.PHONY: bazel-clean
bazel-clean:
	@echo "Cleaning Bazel cache..."
	bazel clean

# ============================================================================
# MLX Service Targets
# ============================================================================

# Run MLX inference service
.PHONY: mlx-run
mlx-run:
	@echo "Starting MLX inference service..."
	python mlx_service/service.py --port 50051

# Run MLX service with model
.PHONY: mlx-run-model
mlx-run-model:
	@echo "Starting MLX inference service with model..."
	python mlx_service/service.py --port 50051 --model meta-llama/Llama-3.1-8B-Instruct

# Install MLX dependencies
.PHONY: mlx-deps
mlx-deps:
	@echo "Installing MLX dependencies..."
	pip install -r mlx_service/requirements.txt

# ============================================================================
# Proto Generation
# ============================================================================

# Generate protobuf code
.PHONY: proto
proto:
	@echo "Generating protobuf code..."
	protoc --proto_path=proto \
		--go_out=go/internal/inference/proto \
		--go_opt=paths=source_relative \
		--go-grpc_out=go/internal/inference/proto \
		--go-grpc_opt=paths=source_relative \
		proto/inference.proto || true
	python -m grpc_tools.protoc \
		--proto_path=proto \
		--python_out=mlx_service \
		--grpc_python_out=mlx_service \
		proto/inference.proto || true

# ============================================================================
# Docker (Go) Targets
# ============================================================================

# Build Go Docker images
.PHONY: go-docker-build
go-docker-build:
	@echo "Building Go Docker images..."
	$(DOCKER_COMPOSE) -f docker-compose.go.yml build

# Start Go services
.PHONY: go-docker-up
go-docker-up:
	@echo "Starting Go services..."
	$(DOCKER_COMPOSE) -f docker-compose.go.yml up -d

# Stop Go services
.PHONY: go-docker-down
go-docker-down:
	@echo "Stopping Go services..."
	$(DOCKER_COMPOSE) -f docker-compose.go.yml down

# View Go service logs
.PHONY: go-docker-logs
go-docker-logs:
	$(DOCKER_COMPOSE) -f docker-compose.go.yml logs -f