{
  description = "mlx-omni-server Nix tooling";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" "x86_64-darwin" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f (import nixpkgs { inherit system; }));
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
    };
}
