# Development Guide

This guide is intended for developers who want to contribute to MLX Omni Server or create their own extensions.

## Setting Up Development Environment

1. Clone the repository:
```bash
git clone https://github.com/madroidmaq/mlx-omni-server.git
cd mlx-omni-server
```

2. Install dependencies using uv:
```bash
uv pip install -e .
```

## Running the Server in Development Mode

There are two ways to run the server during development:

### 1. Using uvicorn (Recommended for development)

```bash
uvicorn mlx_omni_server.main:app --reload --host 0.0.0.0 --port 10240
```

The `--reload` flag enables hot-reload, which automatically restarts the server when code changes are detected. This is particularly useful during development.

### 2. Using the standard entry point

```bash
mlx-omni-server
```


## Contributing Guidelines

1. Fork the repository
2. Create your feature branch:
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. Follow the code style:
   - Use [Black](https://black.readthedocs.io/) for Python code formatting
   - Use [isort](https://pycqa.github.io/isort/) for import sorting
   - Run pre-commit hooks before committing:
     ```bash
     pre-commit install
     pre-commit run --all-files
     ```
4. Write clear commit messages
5. Push to your branch:
   ```bash
   git push origin feature/amazing-feature
   ```
6. Open a Pull Request with:
   - Clear description of the changes
   - Any relevant issue numbers
   - Screenshots for UI changes (if applicable)

## Testing

Run the test suite:
```bash
pytest
```

## Building Documentation

The documentation is written in Markdown and stored in the `docs/` directory.

## Getting Help

- Open an issue for bugs or feature requests
- Join discussions in the GitHub Discussions section
- Check existing issues and pull requests before creating new ones


## Building macOS (arm64) wheelhouse with Nix

To prepare an offline wheelhouse of mlx-omni-server for Apple Silicon Macs using Nix:

With flakes:

```bash
# Enter a shell with python3.11, pip, uv
nix develop

# Build the wheelhouse into ./artifacts
nix run .#wheelhouse
```

Without flakes:

```bash
nix-shell
bash scripts/build_macos_arm64_wheelhouse.sh
```

On macOS (Apple Silicon):

```bash
cd artifacts
bash install_mlx_omni_offline.sh
mlx-omni-server --host 0.0.0.0 --port 10240
```

Notes:
- Targets macosx_12_0_arm64, CPython 3.11 wheels.
- If some dependencies lack macOS wheels, complete installation online on macOS:
  ```bash
  pip install mlx-omni-server==0.5.1
  ```
