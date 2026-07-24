# Native sidecar freezer

Geist is frozen as a PyInstaller **onedir** application. Consumers must stage
the entire generated `geist` directory; copying only the executable is
unsupported. PyInstaller builds are target-native and cannot be cross-compiled.

## Locked build

Use Python 3.11 and the checked-in `uv.lock`. The `packaged` extra supplies the
exact PyInstaller version; it is intentionally declared with the other locked
dependencies rather than installed by this script.

```bash
uv run --frozen --no-dev --extra packaged python scripts/build_native_sidecar.py
```

macOS ARM64 also installs the locked MLX runtime into the frozen environment:

```bash
uv run --frozen --no-dev --extra packaged --extra local-mlx \
  python scripts/build_native_sidecar.py --target darwin-arm64
```

Release builds pass `--codesign-identity` (or set
`GEIST_CODESIGN_IDENTITY`) so PyInstaller signs its executable and collected
MLX libraries before the onedir tree is embedded in a release package. Local
macOS builds may use PyInstaller's default ad-hoc signing.

The command runs the frontend's locked `npm ci`, compiles the React app, and
then creates one of these layouts based on the host:

```text
dist/native/win32-x64/geist/geist.exe
dist/native/darwin-arm64/geist/geist
dist/native/linux-x64/geist/geist
```

Pass `--target <target>` in release automation to assert that a runner has the
expected OS and architecture. It does not enable cross-compilation. When CI has
already compiled `client/geist/build`, pass `--skip-web-build`.

Custom output roots are supported:

```bash
uv run --frozen --no-dev --extra packaged python scripts/build_native_sidecar.py \
  --target linux-x64 \
  --dist-dir /build/geist
```

The complete directory to give a consuming application's stager is
`/build/geist/geist` in that example. Windows and Linux consumers stage their
CPU and Vulkan llama.cpp runtimes separately and configure their locations with
`GEIST_LLAMA_SERVER_PATH` or `GEIST_LLAMA_RUNTIME_ROOT`.

## Included runtime data

Every onedir bundle contains:

- the compiled SPA at `client/geist/build` for same-origin serving;
- the complete Alembic migration tree and `alembic.ini`;
- Geist package metadata and the dynamic database/adapter modules;
- the MLX runner plus MLX native files on macOS ARM64; or
- the `llama-server` runner on Windows/Linux (the llama.cpp runtime remains a
  separately staged sibling directory).

`geist-runtime.json` records the target, backend, tool versions, and SHA-256 of
both dependency lockfiles. It deliberately contains no build timestamp so two
builds can be compared directly. Models are not frozen into the sidecar.

## Release verification

From the generated directory, exercise the actual executable without Python on
`PATH`:

```bash
geist/geist doctor --json
geist/geist serve --host 127.0.0.1 --port 0 --data-dir /tmp/geist-smoke --managed-stdio
```

On Windows use `geist\geist.exe` and a fresh directory under `%TEMP%`. The
managed serve check must print one `geist.ready` JSON line, return a successful
health response from the reported origin, serve the SPA at `/`, and exit after
stdin closes. Windows/Linux release jobs should additionally stage both the CPU
and Vulkan llama.cpp directories. macOS ARM64 release jobs should run an MLX
model download and inference smoke test.
