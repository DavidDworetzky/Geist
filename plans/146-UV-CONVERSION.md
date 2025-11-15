## 146 - UV CONVERSION IMPLEMENTATION PLAN

### Overview
- **Goal**: Migrate project environment and dependency management from Conda to `uv` while keeping Docker, `Makefile`, and local dev flows smooth.
- **Scope**: Python environment management, dependency specification/locking, Docker build process, and developer onboarding/docs.
- **Constraints**:
  - Preserve existing runtime behavior (same Python version, same key dependencies).
  - Avoid breaking CI, `docker compose` workflows, or `make` targets.
  - Prefer minimal extra tooling; rely on `uv` + standard Python where possible.

### Current State Analysis
- **Environment definitions**:
  - `linux_environment.yml`, `linux_environment_x86_x64.yml`, `mac_environment_arm.yml` define Conda envs.
  - `conda-install.sh` bootstraps Conda-based environments.
- **Docker & runtime**:
  - `Dockerfile` and `docker-compose.yml` build and run the backend with Conda.
  - `Makefile` targets (`make build`, `make run`, `make empty`) assume Conda/`environment.yml`-style setup.
  - Backend container name `backend` is used for `docker exec` commands in docs/rules.
- **Python dependencies**:
  - Requirements currently derive from Conda env files (plus manual installs via `conda env export`).
  - No canonical `pyproject.toml` / `requirements.txt` fully owned by `uv` yet.

### Target Architecture with `uv`
- **Environment management**:
  - Use `uv` as the primary tool for:
    - Creating and managing virtual environments.
    - Installing and locking Python dependencies.
  - Replace Conda env files with `uv`-managed lockfiles and minimal metadata.
- **Dependency specification**:
  - Introduce a canonical dependency specification: `pyproject.toml`
  - Keep any non-Python system dependencies documented in `Dockerfile` comments/README.
- **Docker integration**:
  - Docker images install `uv` and use it to:
    - Sync dependencies based on lockfile.
    - Run the app via `uv run` (or activate the `uv`-created venv).
- **Developer workflow**:
  - Local dev uses `uv` commands instead of Conda:
    - `uv venv`, `uv sync`, `uv run`, etc.
    - update `readme.md`
  - Keep `make` targets as the primary entrypoints, but internally switch them to `uv`.

### Phase 1: Inventory and Baseline
1. **Extract canonical dependency list from Conda envs**
   - Parse `linux_environment.yml`, `linux_environment_x86_x64.yml`, and `mac_environment_arm.yml`.
   - Identify:
     - Python version.
     - Core Python packages (and versions) actually used by the project.
     - Any Conda-only/system-level deps that must be replicated via `apt` in Docker.
2. **Verify runtime expectations**
   - Confirm:
     - Minimum Python version we support.
     - CUDA/GPU or ML-specific dependencies required by MLX, vLLM, Sesame, etc.
   - Note any platform-specific packages that may require conditional handling with `uv`.
3. **Create plan file for dependency installs from the root**

### Phase 2: Introduce `uv` Project Definition, Local Development workflow migration and Docker changes.
1. **Add `pyproject.toml`**
   - Define:
     - Project metadata (name, version, description, authors).
     - `requires-python` matching current Conda Python version.
     - `dependencies` list based on extracted inventory (Pin versions where needed; leave others with compatible ranges).
   - Add optional `dev-dependencies` for testing, linting, and tooling (`pytest`, `mypy`, etc.).
2. **Generate `uv` lockfile**
   - Run `uv lock` to create `uv.lock` capturing fully-resolved versions.
   - Verify lockfile includes all required packages for:
     - Backend server.
     - Agents and ML architectures (LLama, Sesame, Moshi, etc.).
     - Tests and migrations.
3. **Add `uv`-specific ignore rules**
   - Ensure lockfiles and `__pycache__` remain handled via `.gitignore` as appropriate.
   - Do **not** ignore `uv.lock` (it should be version-controlled).
4. **Update `readme.md`**
5. **Migrate `conda-install.sh` to similar uv script, and reference from dockerfile**
6. **Update Dockerfile, and any other references to conda install to use uv environments instead.**
    - Install `uv` in the base image
    - Copy `pyproject.toml` and `uv.lock` into the image.
    - Run `uv sync --frozen` during the build to install deps into a project venv
    - Ensure:
        - `PYTHONPATH` is set appropriately
        - `PATH` includes the venv bin directory
7. **Remove `conda-install.sh` and all conda environment solves `linux_environment.*`, `mac_environment.*`**
8. **Make any necessary docker-compose.yml changes**
    -make updates to `uv run pytest` if necessary
9. **Update any CI scripts**
    -Update github CI/CD

### Phase 3: Validation
1. **Functional validation**
   - Run:
     - `docker compose up -d`.
     - Verify no error logs in backend container.
     - `cd /opt/geist && PYTHONPATH=/opt/geist pytest` inside the backend container.
     - `curl` to `localhost:3000` to confirm the app responds.
2. **Cross-platform sanity checks**
   - Validate `uv` setup on:
     - macOS (arm64).
3. **Deprecation of Conda artifacts**
   - Once `uv` path is stable:
     - Make sure these are removed.
       - `linux_environment.yml`, `linux_environment_x86_x64.yml`, `mac_environment_arm.yml`.
       - `conda-install.sh`.
     - Update any remaining references in docs/scripts.


