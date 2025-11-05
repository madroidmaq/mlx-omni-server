# MLX Omni Server Containerization Guide

This guide covers multiple ways to containerize and deploy MLX Omni Server.

## Table of Contents

- [Quick Start](#quick-start)
- [Method 1: Traditional Container Build](#method-1-traditional-container-build)
- [Method 2: Nix-based OCI Images](#method-2-nix-based-oci-images)
- [Running Containers](#running-containers)
- [Configuration](#configuration)
- [Important Notes](#important-notes)

## Quick Start

**Using traditional Docker/Podman:**
```bash
docker build -f Containerfile -t mlx-omni-server:latest .
docker run -p 10240:10240 mlx-omni-server:latest
```

**Using Nix (standalone image):**
```bash
nix build .#oci-image-standalone
docker load < result
docker run -p 10240:10240 mlx-omni-server:0.5.1-standalone
```

## Method 1: Traditional Container Build

### Using Docker

```bash
# Build the image
docker build -f Containerfile -t mlx-omni-server:latest .

# Run the container
docker run -p 10240:10240 mlx-omni-server:latest
```

### Using Podman

```bash
# Build the image
podman build -f Containerfile -t mlx-omni-server:latest .

# Run the container
podman run -p 10240:10240 mlx-omni-server:latest
```

### Using Buildah

```bash
# Build the image
buildah bud -f Containerfile -t mlx-omni-server:latest .

# Export to OCI format
buildah push mlx-omni-server:latest oci:mlx-omni-server:latest
```

## Method 2: Nix-based OCI Images

Nix provides reproducible, declarative container builds. Two image types are available:

### Image Type 1: oci-image (Development)

Best for development with live source code updates.

**Features:**
- Streamable image format (efficient, no tar overhead)
- Requires mounting source code as volume
- Dependencies installed on first run

**Build:**
```bash
nix build .#oci-image
```

**Load into Docker:**
```bash
./result | docker load
```

**Run:**
```bash
docker run -v $(pwd):/app -p 10240:10240 mlx-omni-server:0.5.1
```

### Image Type 2: oci-image-standalone (Distribution)

Best for distribution and deployment.

**Features:**
- Self-contained with source code included
- No volume mount required
- Ready for distribution

**Build:**
```bash
nix build .#oci-image-standalone
```

**Load into Docker:**
```bash
docker load < result
```

**Run:**
```bash
docker run -p 10240:10240 mlx-omni-server:0.5.1-standalone
```

### Cross-Platform Builds with Nix

Build Linux images on macOS or other platforms:

```bash
# Build for ARM64 Linux (e.g., Raspberry Pi, AWS Graviton)
nix build .#packages.aarch64-linux.oci-image-standalone

# Build for x86_64 Linux (most cloud servers)
nix build .#packages.x86_64-linux.oci-image-standalone
```

## Running Containers

### Basic Usage

```bash
docker run -p 10240:10240 mlx-omni-server:latest
```

### With Environment Variables

```bash
docker run \
  -e MLX_OMNI_PORT=8080 \
  -e MLX_OMNI_LOG_LEVEL=debug \
  -p 8080:8080 \
  mlx-omni-server:latest
```

### With Persistent Storage

Mount volumes for models and logs:

```bash
docker run \
  -v ./models:/models \
  -v ./logs:/logs \
  -p 10240:10240 \
  mlx-omni-server:latest
```

### Interactive Mode

```bash
docker run -it --rm \
  -p 10240:10240 \
  mlx-omni-server:latest \
  bash
```

### Docker Compose

Create a `docker-compose.yml`:

```yaml
version: '3.8'

services:
  mlx-omni-server:
    image: mlx-omni-server:latest
    ports:
      - "10240:10240"
    environment:
      - MLX_OMNI_LOG_LEVEL=info
      - HF_HOME=/models
    volumes:
      - ./models:/models
      - ./logs:/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:10240/health"]
      interval: 30s
      timeout: 5s
      retries: 5
```

Run with:
```bash
docker-compose up -d
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MLX_OMNI_PORT` | `10240` | Server port |
| `MLX_OMNI_LOG_LEVEL` | `info` | Log level (debug, info, warning, error) |
| `HF_HOME` | `/models` | HuggingFace cache directory |
| `PYTHONDONTWRITEBYTECODE` | `1` | Prevent Python from writing .pyc files |
| `PYTHONUNBUFFERED` | `1` | Force Python stdout/stderr to be unbuffered |

### Volume Mounts

| Mount Point | Purpose |
|-------------|---------|
| `/models` | HuggingFace model cache (persistent storage) |
| `/logs` | Application logs |
| `/app` | Application source (dev image only) |

### Exposed Ports

- `10240/tcp` - Default server port

## Important Notes

### MLX and Metal Acceleration

MLX is optimized for Apple Silicon (M1/M2/M3/M4) with Metal acceleration. When running in a container:

- **Linux containers**: Will fall back to CPU-only execution
- **No GPU acceleration**: Metal is macOS-specific and unavailable in containers
- **Performance**: Expect reduced performance compared to native macOS execution

### Recommendations

1. **Development on macOS**: Use native installation (`pip install mlx-omni-server`)
2. **Linux deployment**: Use containers for consistency, but expect CPU-only operation
3. **Testing**: Containers are useful for CI/CD and reproducible environments

### First Run

Both Nix images install Python dependencies on first container start:

```bash
docker run mlx-omni-server:0.5.1-standalone
# Output:
# First run: Installing mlx-omni-server...
# Installing dependencies...
# Starting MLX Omni Server on port 10240...
```

Subsequent runs skip installation and start immediately.

### Network Requirements

- First container start requires internet access to download Python packages
- Model downloads require internet (unless pre-cached in `/models` volume)

## Advanced Usage

### Building for Specific Architectures

```bash
# Docker BuildX for multi-platform
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -f Containerfile \
  -t mlx-omni-server:latest .
```

### Exporting Images

```bash
# Save to tar file
docker save mlx-omni-server:latest -o mlx-omni-server.tar

# Load on another machine
docker load -i mlx-omni-server.tar
```

### Registry Push

```bash
# Tag for registry
docker tag mlx-omni-server:latest your-registry/mlx-omni-server:0.5.1

# Push to registry
docker push your-registry/mlx-omni-server:0.5.1
```

## Troubleshooting

### Container won't start

Check logs:
```bash
docker logs <container-id>
```

### Permission errors

Ensure model/log directories are writable:
```bash
mkdir -p models logs
chmod 777 models logs
```

### Network issues

Verify container networking:
```bash
docker run --rm -it mlx-omni-server:latest curl http://localhost:10240/health
```

### Debugging

Run with shell access:
```bash
docker run --rm -it --entrypoint bash mlx-omni-server:latest
```

## See Also

- [README.md](README.md) - Main project documentation
- [Containerfile](Containerfile) - Container build definition
- [flake.nix](flake.nix) - Nix build configuration
- [OpenAI API Guide](docs/openai-api.md)
- [Anthropic API Guide](docs/anthropic-api.md)

## License

MIT License - See [LICENSE](LICENSE) for details