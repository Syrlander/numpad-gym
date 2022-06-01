# Numpad Gym
The package contained in this repo includes 3 different versions of the nummpad envrionment described in [1]. The package can be installed by cloning this repository and then installing via pip as:
```
git clone https://github.com/Syrlander/numpad-gym
cd numpad-gym
pip install .
```
## General Environment description
The Numpad environment consists of $N^2$ tiles structured in a square grid. We call $N$ the numpad size. The agent controls a ball that can be rolled between tiles. At initialization, the environment selects a random sequence of $n$ directly neighboring tiles\footnote{Excluding diagonally neighboring tiles.} $S$, with $n \leq N^2$, and tiles only occurring once in the sequence. As such, the ball can always roll from one tile in the sequence directly onto the next, without touching any other tiles. 

The agent's task is to make the ball press each tile of the sequence in the correct order, if the ball presses the wrong next tile the progress of correctly pressed tiles is reset and the ball has to start over again. A tile is considered pressed as soon as the ball touches it. When the correct tile is pressed it will "light up", which will be visible in the following observation. Tiles will keep their light on as long as the agent does not press a wrong tile. The agent receives a reward of $+1$ the first time it presses the correct next tile in the sequence. Meaning if it presses the wrong tile and has to start over, it will not get any reward before it discovers a new tile in the sequence. After pressing the last tile in the sequence, the agent can go back to the first tile and run the sequence again, and will again receive a reward for each correct tile, as if it pressed them for the first time.

When the agent presses the last tile in the sequence, the light will turn on in the last tile. In the next step, if the tile it touches is the first in the sequence, all tiles except the first in the sequence will turn off, otherwise, all the tiles will turn off. An episode ends after a specified number of time steps, so the agent has a limited time to complete the sequence.

As in [1] it is possible to use task cues during training. These consist of a random subset of tiles in the sequence lighting up in the first observation of the episode.

As a slight deviation from the original implementations, our implementation of the environment does not provide a "jump" action, allowing the agent to move over tiles without pressing them.

## Numpad Discrete
In the discrete case the action space consists of four actions, corresponding to each direction the ball can roll (up, down, left, right). When taking an action the ball will move one tile in the specified direction. The ball is always placed on exactly one tile. If the ball cannot move any further in the direction specified by the action, the ball will not move and the lights reset.

The observation consists of two $N \times N$ matrices, the first matrix is a binary map of light on/off, the other matrix is a binary map for the presence of the ball on each tile.

## Numpad Continuous
In the continuous version, the action space is a vector in 2 dimensions, denoting the acceleration to apply to the ball. In each step, the acceleration is added to the ball's current velocity, and the ball is moved in the direction and distance specified by its velocity. If the ball hits the boundary of the environment, the ball's velocity in the direction from which it hit the boundary is set to 0. That is, if it hits the bottom or top boundary, the velocity in the y-direction is set to 0 and 0 in the x-direction when hitting the left and right boundary. 

As opposed to the discrete case the ball is not necessarily touching a tile at all times, because the environment can be configured to have some spacing between each tile. Touching the  spacing always yields 0 reward, but does not reset the sequence. A tile is pressed once the center of the ball touches the tile\footnote{Performing collision detection between the ball and tiles using the ball center, it should be noted that the ball can visually appear to be touching a tile during a rendering since the ball is rendered as a square for development simplicity.}. The size of the spacing, tiles, and ball are all parameters given to the environment and can be varied.  

In this environment, we have multiple different modes of observation.
  * RGB image, the size of which depends on the parameters of the environment.
  * Greyscale image, in the same size as the RGB image.
  * RAM input given as follows: 
  $$(ball_x, ball_y, velocity_x, velocity_y) \circ (1 \text{ if } \text{ligthIsOn}(t) \text{ else } 0 \text{ for all } t \text{ in tiles})$$
  Where $\circ$ denotes tuple concatenation, $(ball_x, ball_y)$ are the normalized coordinates of the ball, so that $(0,0)$ is the upper left corner, and $(1,1)$ is the lower right corner. $(velocity_x, velocity_y)$ is the velocity of the ball normalized so that $(-1,1)$ corresponds to the ball moving left and down at the maximum speed allowed, $(1, 0)$ corresponds to moving right at the maximum allowed speed and not moving on the other axis. \newpage \noindent The maximum speed in any one direction is set to half the width of each tile. The tuple on the right side is simply a tuple with a 1 for all the tiles where the light is on, and a zero for all the other tiles. We normalize the position to be between 0 and 1 and velocity to be between -1 and 1, so that all inputs to the model are on a similar scale, instead of having the position be between 0 and the pixel width of the generated image.

## Using the environments
The envrionments can be registred in Open AI's Gym [2] using 
```python
gym.envs.register(
        "numpad_discrete-v1",
        entry_point="numpad_gym.numpad_discrete:Environment",
    )
```
Each environment takes a config-object as argument to its constructor eg.:
```python
import numpad_gym
import gym

conf = numpad_gym.numpad_discrete.Config()
env = numpad_gym.numpad_discrete.Environment(conf)

# Or, if the environment is registred as shown above:
conf = numpad_gym.numpad_discrete.Config()
env = gym.make("numpad_discrete-v1", config=conf)
```

# References
[1] Jan Humplik et al. “Meta reinforcement learning as task inference”. In: CoRR abs/1905.06424 (2019). arXiv: 1905.06424. url: http://arxiv.org/abs/1905.06424 \
[2] Greg Brockman et al. OpenAI Gym. 2016. eprint: arXiv:1606.01540.
