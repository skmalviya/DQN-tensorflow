## For installing openai/baselines

1. sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
2. pip install virtualenv
3. virtualenv --python=python3 openai_env
4. source openai_env/bin/activate
5. pip install tensorflow==1.14
6. git clone https://github.com/openai/baselines.git
7. cd baselines
8. python setup.py install
9. Followed link to install https://github.com/openai/mujoco-py
    * Download the MuJoCo version 2.0 binaries for Linux or OSX.
    * Unzip the downloaded mujoco200 directory into ~/.mujoco/mujoco200
    * write https://github.com/MahanFathi/iLQG-MuJoCo/blob/master/bin/mjkey.txt to /home/.mujoco/
    * export LD_LIBRARY_PATH=/home/shrikant/.mujoco/mujoco200/bin:$LD_LIBRARY_PATH
    * sudo apt-get install libglew-dev
    * sudo apt-get install patchelf
    * sudo apt install libosmesa6-dev libosmesa6 libglapi-mesa
    * sudo apt-file search "GL/osmesa.h"
    * sudo apt install --reinstall build-essential & cmake
    * sudo apt install --reinstall libgl1-mesa-dev
    * (may not be required.)sudo apt install --reinstall libgl1-mesa-glx
    * sudo ln -s /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/libGL.so
    * pip install -U 'mujoco-py<2.1,>=2.0'
10. pip install pytest
11. python -m baselines.run --alg=deepq --env=PongNoFrameskip-v4 --num_timesteps=1e6


Exploring : python -m baselines.run --alg=deepq --env=PongNoFrameskip-v4 --num_timesteps=1e6
## Changes made in the code to run it on python3.6.9
If you dont have gpu:
```
--use_gpu=False
```

* agent.py
```
@ line : 227,376 : xrange --> range
# NameError: name 'reduce' is not defined :
from functools import reduce
# TypeError: unsupported operand type(s) for +: 'dict_values' and 'list' :
@ line 328 : self.w.values() --> list(self.w.values())

```
* environment.py
```
@ line : 23,73 : xrange --> range
# AssertionError: Cannot call env.step() before calling reset():
@ line 16 : self._screen = None --> self._screen = self.env.reset()
```


# Human-Level Control through Deep Reinforcement Learning

Tensorflow implementation of [Human-Level Control through Deep Reinforcement Learning](http://home.uchicago.edu/~arij/journalclub/papers/2015_Mnih_et_al.pdf).

![model](assets/model.png)

This implementation contains:

1. Deep Q-network and Q-learning
2. Experience replay memory
    - to reduce the correlations between consecutive updates
3. Network for Q-learning targets are fixed for intervals
    - to reduce the correlations between target and predicted Q-values


## Requirements

- Python 2.7 or Python 3.3+
- [gym](https://github.com/openai/gym)
- [tqdm](https://github.com/tqdm/tqdm)
- [SciPy](http://www.scipy.org/install.html) or [OpenCV2](http://opencv.org/)
- [TensorFlow 0.12.0](https://github.com/tensorflow/tensorflow/tree/r0.12)


## Usage

First, install prerequisites with:

    $ pip install tqdm gym[all]

To train a model for Breakout:

    $ python main.py --env_name=Breakout-v0 --is_train=True
    $ python main.py --env_name=Breakout-v0 --is_train=True --display=True

To test and record the screen with gym:

    $ python main.py --is_train=False
    $ python main.py --is_train=False --display=True


## Results

Result of training for 24 hours using GTX 980 ti.

![best](assets/best.gif)


## Simple Results

Details of `Breakout` with model `m2`(red) for 30 hours using GTX 980 Ti.

![tensorboard](assets/0620_scalar_step_m2.png)

Details of `Breakout` with model `m3`(red) for 30 hours using GTX 980 Ti.

![tensorboard](assets/0620_scalar_step_m3.png)


## Detailed Results

**[1] Action-repeat (frame-skip) of 1, 2, and 4 without learning rate decay**

![A1_A2_A4_0.00025lr](assets/A1_A2_A4_0.00025lr.png)

**[2] Action-repeat (frame-skip) of 1, 2, and 4 with learning rate decay**

![A1_A2_A4_0.0025lr](assets/A1_A2_A4_0.0025lr.png)

**[1] & [2]**

![A1_A2_A4_0.00025lr_0.0025lr](assets/A1_A2_A4_0.00025lr_0.0025lr.png)


**[3] Action-repeat of 4 for DQN (dark blue) Dueling DQN (dark green) DDQN (brown) Dueling DDQN (turquoise)**

The current hyper parameters and gradient clipping are not implemented as it is in the paper.

![A4_duel_double](assets/A4_duel_double.png)


**[4] Distributed action-repeat (frame-skip) of 1 without learning rate decay**

![A1_0.00025lr_distributed](assets/A4_0.00025lr_distributed.png)

**[5] Distributed action-repeat (frame-skip) of 4 without learning rate decay**

![A4_0.00025lr_distributed](assets/A4_0.00025lr_distributed.png)


## References

- [simple_dqn](https://github.com/tambetm/simple_dqn.git)
- [Code for Human-level control through deep reinforcement learning](https://sites.google.com/a/deepmind.com/dqn/)


## License

MIT License.
