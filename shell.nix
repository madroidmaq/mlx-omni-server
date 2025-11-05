let
  pkgs = import <nixpkgs> {};
in pkgs.mkShell {
  packages = [
    pkgs.python311
    pkgs.python311Packages.pip
    pkgs.uv
    pkgs.curl
    pkgs.git
    pkgs.bash
    pkgs.coreutils
  ];
  shellHook = ''
    echo "Nix shell ready: python3.11, pip, uv, curl, git"
  '';
}
