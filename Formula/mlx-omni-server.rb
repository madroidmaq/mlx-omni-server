class MlxOmniServer < Formula
  include Language::Python::Virtualenv

  desc "OpenAI-compatible API server powered by Apple's MLX framework"
  homepage "https://github.com/madroidmaq/mlx-omni-server"
  url "https://github.com/madroidmaq/mlx-omni-server/archive/refs/tags/v0.4.3.tar.gz"
  # To get the actual SHA256 hash, download the package and run:
  # curl -L -o mlx-omni-server-0.4.3.tar.gz https://github.com/madroidmaq/mlx-omni-server/archive/refs/tags/v0.4.3.tar.gz
  # shasum -a 256 mlx-omni-server-0.4.3.tar.gz
  sha256 "8d39e28aa9249f102accfb8a9f09d44adf1ca2738b1338335cfd2622796a88b6"
  license "MIT"

  depends_on "python@3.11"
  depends_on :macos

  # This formula installs the following key dependencies from PyPI:
  # - fastapi: Web framework for building APIs
  # - uvicorn: ASGI server
  # - pydantic: Data validation
  # - mlx-lm: Language model support using MLX
  # - mlx-whisper: Speech-to-text using MLX
  # - f5-tts-mlx: Text-to-speech using MLX
  # - mflux: Image generation using MLX
  # - mlx-embeddings: Text embeddings using MLX

  def install
    # Install all dependencies from pyproject.toml
    system "pip3", "install", "--prefix=#{libexec}", "."
    bin.install Dir["#{libexec}/bin/*"]
    bin.env_script_all_files(libexec/"bin", PATH: "#{libexec}/bin:$PATH", PYTHONPATH: "#{libexec}/lib/python3.11/site-packages")
  end

  # Only allow installation on Apple Silicon Macs
  pour_bottle? do
    reason "MLX Omni Server requires Apple Silicon (M-series) chips"
    satisfy { Hardware::CPU.arm? }
  end

  def caveats
    <<~EOS
      MLX Omni Server is designed specifically for Apple Silicon (M-series) chips.
      It will not work on Intel-based Macs.

      To start the server:
        mlx-omni-server

      The server will be available at:
        http://localhost:10240
    EOS
  end

  test do
    # Add a basic test to verify the installation
    assert_match "MLX Omni Server", shell_output("#{bin}/mlx-omni-server --help")
  end
end
