# neoworld

## DinoWM CKPT

```bash
wget https://osf.io/xvzs4/download -O ckpt.zip && unzip ckpt.zip && rm ckpt.zip
```

## DinWM PushT Dataset

```bash
wget https://osf.io/k2d8w/download -O dinowm_pushT.zip && unzip dinowm_pushT.zip && rm dinowm_pushT.zip
```

## TODO:

- [ ] clean code
- [ ] add more documentation
- [ ] make everything uniform, e.g solver param names
- [ ] make transform more clear (e.g wrapper for world model, wrapper for planning etc)

## Installation

Follow the below isntruction (if you are not using conda, adjust accordingly)

1. create conda env
  ```
  conda create -n xeno python=3.10
  ```

2. activate your env and install uv (faster package manager)
  ```
  conda activate xeno
  pip3 install uv
  ```
3. install our package
  ```
  uv pip install -e .
  ```
4. install MUJOCO
  ```
  wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
  tar -xvf mujoco210-linux-x86_64.tar.gz -C ~/.mujoco/
  ```
  and add to your `.bashrc`
  ```
  export MUJOCO_HOME="$HOME/.mujoco/mujoco210"
export PATH="$MUJOCO_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$MUJOCO_HOME/bin:$LD_LIBRARY_PATH"
```



