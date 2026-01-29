#!/usr/bin/env bash
set -eo pipefail   # no -u because of conda/openjdk hooks

### CONFIGURATION
ENV_NAME="minerl-env"
PYTHON_VERSION="3.10"
UV_VERSION=""
GYM_SRC_DIR="$HOME/src-gym019"
MINERL_SRC_DIR="$HOME/src-minerl"

### 0. CHECK THAT CONDA EXISTS
if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH. Please install Miniconda/Anaconda first."
  exit 1
fi

### CLEAN PREVIOUS ENV IF PRESENT
conda deactivate 2>/dev/null || true
if conda env list | grep -qE "^[^#]*\b${ENV_NAME}\b"; then
  echo "Removing existing env '${ENV_NAME}'..."
  conda env remove -n "$ENV_NAME" -y
fi

### 1. CREATE AND ACTIVATE CONDA ENV
conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

### 2. INSTALL OPENJDK 8 (via conda-forge)
conda install -y -c conda-forge openjdk=8

# Ensure JAVA_HOME points to this JDK (helps Gradle)
export JAVA_HOME="$CONDA_PREFIX"
export PATH="$JAVA_HOME/bin:$PATH"

### 3. INSTALL uv INSIDE THE ENV
python -m pip install --upgrade pip
if [[ -n "$UV_VERSION" ]]; then
  pip install "uv[python]==$UV_VERSION"
else
  pip install "uv[python]"
fi

### 4. INSTALL PYTORCH (GPU, LATEST) WITH uv
# This uses the PyPI "default" (CUDA-enabled) wheels for your platform.
uv pip install --python "$(which python)" torch torchvision torchaudio

### 5. BASIC DS/AI STACK WITH uv
uv pip install --python "$(which python)" \
  numpy \
  pandas \
  matplotlib \
  jupyterlab \
  ipykernel \
  tqdm \
  gymnasium \
  stable-baselines3 \
  opencv-python

###############################################################################
### 6. DOWNLOAD, PATCH, AND INSTALL GYM 0.19.0 FROM LOCAL SOURCE
###############################################################################
echo "Preparing local gym==0.19.0 source in $GYM_SRC_DIR ..."
mkdir -p "$GYM_SRC_DIR"
cd "$GYM_SRC_DIR"

# Download source tarball if not already present
if [[ ! -f gym-0.19.0.tar.gz ]]; then
  curl -L -o gym-0.19.0.tar.gz \
    https://files.pythonhosted.org/packages/source/g/gym/gym-0.19.0.tar.gz
fi

# Extract fresh copy
rm -rf gym-0.19.0
tar xf gym-0.19.0.tar.gz
cd gym-0.19.0

# Replace setup.py with minimal, modern-friendly version
mv setup.py setup.py.orig

cat > setup.py << 'EOF'
import os.path
import sys
from setuptools import find_packages, setup

# Don't import gym module here, since deps may not be installed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "gym"))
from version import VERSION

setup(
    name="gym",
    version=VERSION,
    description="The OpenAI Gym: A toolkit for developing and comparing your reinforcement learning agents.",
    url="https://github.com/openai/gym",
    author="OpenAI",
    author_email="gym@openai.com",
    license="",
    packages=[package for package in find_packages() if package.startswith("gym")],
    zip_safe=False,
    install_requires=[
        "numpy>=1.18.0",
        "cloudpickle>=1.2.0,<1.7.0",
    ],
    extras_require={},  # no extras to avoid metadata issues
    package_data={
        "gym": [
            "envs/mujoco/assets/*.xml",
            "envs/classic_control/assets/*.png",
            "envs/robotics/assets/LICENSE.md",
            "envs/robotics/assets/fetch/*.xml",
            "envs/robotics/assets/hand/*.xml",
            "envs/robotics/assets/stls/fetch/*.stl",
            "envs/robotics/assets/stls/hand/*.stl",
            "envs/robotics/assets/textures/*.png",
        ]
    },
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
EOF

# Install patched gym
python -m pip install .
python - << 'EOF'
import gym
print("Installed gym version:", gym.__version__)
EOF


#!/bin/bash
set -e

echo "=== MineRL Installation Script ==="

REPO_URL="https://github.com/danijar/minerl.git"
COMMIT="87ca0e3"

# Clone
echo "[1/4] Cloning MineRL repository..."
if [ -d "$MINERL_SRC_DIR" ]; then
    rm -rf "$MINERL_SRC_DIR"
fi
git clone "$REPO_URL" "$MINERL_SRC_DIR"
cd "$MINERL_SRC_DIR"
git checkout "$COMMIT"

# Patch build.gradle - fix MixinGradle dependency
echo "[2/4] Patching build.gradle..."
BUILD_GRADLE="minerl/Malmo/Minecraft/build.gradle"
sed -i.bak "s|com.github.SpongePowered:MixinGradle:dcfaf61|org.spongepowered:mixingradle:0.6-SNAPSHOT|g" "$BUILD_GRADLE"
# Add SpongePowered repo (insert after jitpack line)
sed -i.bak "/maven { url 'https:\/\/jitpack.io' }/a\\        maven { url 'https://repo.spongepowered.org/repository/maven-public/' }" "$BUILD_GRADLE"
rm -f "${BUILD_GRADLE}.bak"

# Install
echo "[4/4] Installing..."
uv pip install pyglet==1.5.27
pip install -e .

echo "=== Done ==="

### 9. CREATE A RENDER TEST SCRIPT IN CURRENT DIR
cd ~/
cat > minerl_render_test.py << 'EOF'
import gym
import minerl
import minerl.herobraine.envs  # Triggers registration

def main():
    env = gym.make("MineRLBasaltBuildVillageHouse-v0")
    obs = env.reset()
    done = False
    step = 0
    while not done and step < 200:
        ac = env.action_space.noop()
        ac["camera"] = [0, 3]
        obs, reward, done, info = env.step(ac)
        a = env.render()
        print(a.shape, a.min(),a.max())
        step += 1
    env.close()
    print("Render test completed.")

if __name__ == "__main__":
    main()
EOF

### 10. SUMMARY & HOW TO RUN
echo
echo "===== ENVIRONMENT READY ====="
echo "Conda env: $ENV_NAME"
echo "Python: $(python --version)"
echo "Java:"
java -version 2>&1 | sed 's/^/  /'
echo "gym version:"
python -c "import gym; print('  gym', gym.__version__)"
echo "Minerl version:"
python -c "import minerl; print('  minerl', minerl)"
echo
echo "To use this environment:"
echo "  conda activate $ENV_NAME"
echo
echo "To test headless rendering with existing xvfb-run:"
echo "  conda activate $ENV_NAME"
echo '  xvfb-run -s "-screen 0 1400x900x24" python ~/minerl_render_test.py'
echo
echo "If you have a DISPLAY already set, you can also try:"
echo "  conda activate $ENV_NAME"
echo "  python minerl_render_test.py"