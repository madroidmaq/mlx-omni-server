# Homebrew Formula for MLX Omni Server

This document provides instructions for testing and submitting the MLX Omni Server formula to Homebrew.

## Prerequisites

- Homebrew installed on your Mac
- Apple Silicon Mac (M-series chip)
- Git

## Testing the Formula Locally

1. **Get the actual SHA256 hash for the package**:

   ```bash
   curl -L -o mlx-omni-server-0.4.3.tar.gz https://github.com/madroidmaq/mlx-omni-server/archive/refs/tags/v0.4.3.tar.gz
   shasum -a 256 mlx-omni-server-0.4.3.tar.gz
   ```

   Update the `sha256` value in the formula with the actual hash.

2. **Install the formula locally**:

   ```bash
   brew install --build-from-source ./mlx-omni-server.rb
   ```

3. **Test the installation**:

   ```bash
   mlx-omni-server --help
   ```

   You should see the help message for MLX Omni Server.

## Submitting to Homebrew

There are two main ways to submit your formula to Homebrew:

### Option 1: Submit to Homebrew Core (Official Repository)

1. **Fork the Homebrew Core repository**:

   ```bash
   brew tap homebrew/core
   cd "$(brew --repository homebrew/core)"
   git remote add me https://github.com/YOUR_USERNAME/homebrew-core
   git checkout -b mlx-omni-server
   ```

2. **Copy your formula to the Formula directory**:

   ```bash
   cp /path/to/mlx-omni-server.rb Formula/m/mlx-omni-server.rb
   ```

3. **Test the formula**:

   ```bash
   brew audit --new-formula Formula/m/mlx-omni-server.rb
   brew style Formula/m/mlx-omni-server.rb
   brew install --build-from-source Formula/m/mlx-omni-server.rb
   brew test Formula/m/mlx-omni-server.rb
   ```

4. **Commit and push your changes**:

   ```bash
   git add Formula/m/mlx-omni-server.rb
   git commit -m "mlx-omni-server: new formula"
   git push me mlx-omni-server
   ```

5. **Create a pull request** on the [Homebrew Core repository](https://github.com/Homebrew/homebrew-core).

### Option 2: Create Your Own Tap (Recommended for Project-Specific Formulas)

1. **Create a new GitHub repository** named `homebrew-tap` (or any name you prefer).

2. **Add your formula to the repository**:

   ```bash
   mkdir -p Formula
   cp mlx-omni-server.rb Formula/
   git add Formula/mlx-omni-server.rb
   git commit -m "Add mlx-omni-server formula"
   git push
   ```

3. **Tap your repository**:

   ```bash
   brew tap YOUR_USERNAME/tap
   ```

4. **Install your formula**:

   ```bash
   brew install YOUR_USERNAME/tap/mlx-omni-server
   ```

## Additional Resources

- [Homebrew Formula Cookbook](https://docs.brew.sh/Formula-Cookbook)
- [Homebrew Python for Formula Authors](https://docs.brew.sh/Python-for-Formula-Authors)
- [Homebrew Acceptable Formulae](https://docs.brew.sh/Acceptable-Formulae)
