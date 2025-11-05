{
  description = "mlx-omni-server Nix tooling";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" "x86_64-darwin" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f (import nixpkgs { inherit system; }));

      # Version from pyproject.toml
      version = "0.5.1";
    in {
      devShells = forAllSystems (pkgs: with pkgs; {
        default = mkShell {
          packages = [
            python311
            python311Packages.pip
            uv
            curl
            git
            bash
            coreutils
          ];
          shellHook = ''
            echo "Nix shell ready: python3.11, pip, uv, curl, git"
          '';
        };
      });

      apps = forAllSystems (pkgs:
        let
          pkgVersion = builtins.getEnv "PKG_VERSION";
          version = if pkgVersion != "" then pkgVersion else "0.5.1";
        in {
          default = {
            type = "app";
            program = "${pkgs.writeShellScript "run-server" ''
              set -euo pipefail
              export PATH="${pkgs.python311}/bin:${pkgs.uv}/bin:$PATH"

              # Install dependencies if not already installed
              if [ ! -d ".venv" ]; then
                echo "Creating virtual environment and installing dependencies..."
                ${pkgs.uv}/bin/uv venv
                ${pkgs.uv}/bin/uv pip install -e .
              fi

              # Activate venv and run server
              source .venv/bin/activate
              echo "Starting MLX Omni Server on port 10240..."
              python -m mlx_omni_server.main --port 10240 "$@"
            ''}";
          };

          wheelhouse = {
            type = "app";
            program = "${pkgs.writeShellScript "build-wheelhouse" ''
              set -euo pipefail
              export PKG_VERSION="${version}"
              bash scripts/build_macos_arm64_wheelhouse.sh
            ''}";
          };
        });

      # OCI container images (Linux only)
      packages = nixpkgs.lib.genAttrs [ "x86_64-linux" "aarch64-linux" ] (system:
        let
          pkgs = import nixpkgs { inherit system; };
        in {
          # Streamable OCI image - recommended for production use
          # Build with: nix build .#oci-image
          # Load with: nix build .#oci-image && ./result | docker load
          oci-image = pkgs.dockerTools.streamLayeredImage {
            name = "mlx-omni-server";
            tag = version;

            contents = with pkgs; [
              # Base utilities
              bash
              coreutils
              curl
              cacert
              git

              # Python and package manager
              python311
              uv
            ];

            config = {
              Env = [
                "PYTHONDONTWRITEBYTECODE=1"
                "PYTHONUNBUFFERED=1"
                "PIP_NO_CACHE_DIR=1"
                "HF_HOME=/models"
                "MLX_OMNI_LOG_LEVEL=info"
                "MLX_OMNI_PORT=10240"
                "PATH=/usr/local/bin:/usr/bin:/bin:${pkgs.python311}/bin:${pkgs.uv}/bin"
                "SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
              ];

              WorkingDir = "/app";

              ExposedPorts = {
                "10240/tcp" = {};
              };

              # Install the package at runtime (first start)
              # This is done because building Python packages with complex ML deps
              # in Nix is challenging. The image will install on first run.
              Cmd = [ "${pkgs.bash}/bin/bash" "-c" ''
                # Copy source files if they don't exist
                if [ ! -f /app/pyproject.toml ]; then
                  echo "Error: /app must contain the mlx-omni-server source code"
                  echo "Mount the source with: docker run -v $(pwd):/app ..."
                  exit 1
                fi

                # Install if not already installed
                if ! command -v mlx-omni-server >/dev/null 2>&1; then
                  echo "Installing mlx-omni-server..."
                  ${pkgs.uv}/bin/uv pip install --system --no-cache .
                fi

                # Create model and log directories
                mkdir -p /models /logs

                # Start the server
                echo "Starting MLX Omni Server on port ''${MLX_OMNI_PORT}..."
                exec mlx-omni-server --host 0.0.0.0 --port "''${MLX_OMNI_PORT}"
              '' ];
            };
          };

          # Self-contained image with source included
          # Build with: nix build .#oci-image-standalone
          # Load with: nix build .#oci-image-standalone && docker load < result
          # Note: Installs Python packages on first run (requires network)
          oci-image-standalone = pkgs.dockerTools.buildLayeredImage {
            name = "mlx-omni-server";
            tag = "${version}-standalone";

            contents = with pkgs; [
              bash
              coreutils
              curl
              cacert
              git
              python311
              uv
            ];

            extraCommands = ''
              # Create app directory and copy source
              mkdir -p app
              cp -r ${self}/src app/
              cp ${self}/pyproject.toml ${self}/README.md ${self}/LICENSE app/

              # Create model and log directories
              mkdir -p models logs
              chmod 777 models logs
            '';

            config = {
              Env = [
                "PYTHONDONTWRITEBYTECODE=1"
                "PYTHONUNBUFFERED=1"
                "PIP_NO_CACHE_DIR=1"
                "HF_HOME=/models"
                "MLX_OMNI_LOG_LEVEL=info"
                "MLX_OMNI_PORT=10240"
                "SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
              ];

              WorkingDir = "/app";

              ExposedPorts = {
                "10240/tcp" = {};
              };

              Cmd = [ "${pkgs.bash}/bin/bash" "-c" ''
                # Install if not already installed (first run)
                if ! command -v mlx-omni-server >/dev/null 2>&1; then
                  echo "First run: Installing mlx-omni-server..."
                  ${pkgs.uv}/bin/uv pip install --system --no-cache .
                fi

                # Create directories
                mkdir -p /models /logs

                # Start server
                exec mlx-omni-server --host 0.0.0.0 --port "''${MLX_OMNI_PORT}"
              '' ];
            };
          };
        }
      );
    };
}
