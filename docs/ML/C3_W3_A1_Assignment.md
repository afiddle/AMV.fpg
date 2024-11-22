# Deep Q-Learning - Lunar Lander

In this assignment, you will train an agent to land a lunar lander safely on a landing pad on the surface of the moon.


# Outline
- [ 1 - Import Packages <img align="Right" src="./images/lunar_lander.gif" width = 60% >](#1)
- [ 2 - Hyperparameters](#2)
- [ 3 - The Lunar Lander Environment](#3)
  - [ 3.1 Action Space](#3.1)
  - [ 3.2 Observation Space](#3.2)
  - [ 3.3 Rewards](#3.3)
  - [ 3.4 Episode Termination](#3.4)
- [ 4 - Load the Environment](#4)
- [ 5 - Interacting with the Gym Environment](#5)
    - [ 5.1 Exploring the Environment's Dynamics](#5.1)
- [ 6 - Deep Q-Learning](#6)
  - [ 6.1 Target Network](#6.1)
    - [ Exercise 1](#ex01)
  - [ 6.2 Experience Replay](#6.2)
- [ 7 - Deep Q-Learning Algorithm with Experience Replay](#7)
  - [ Exercise 2](#ex02)
- [ 8 - Update the Network Weights](#8)
- [ 9 - Train the Agent](#9)
- [ 10 - See the Trained Agent In Action](#10)
- [ 11 - Congratulations!](#11)
- [ 12 - References](#12)


_**NOTE:** To prevent errors from the autograder, you are not allowed to edit or delete non-graded cells in this lab. Please also refrain from adding any new cells. 
**Once you have passed this assignment** and want to experiment with any of the non-graded code, you may follow the instructions at the bottom of this notebook._

<a name="1"></a>
## 1 - Import Packages

We'll make use of the following packages:
- `numpy` is a package for scientific computing in python.
- `deque` will be our data structure for our memory buffer.
- `namedtuple` will be used to store the experience tuples.
- The `gym` toolkit is a collection of environments that can be used to test reinforcement learning algorithms. We should note that in this notebook we are using `gym` version `0.24.0`.
- `PIL.Image` and `pyvirtualdisplay` are needed to render the Lunar Lander environment.
- We will use several modules from the `tensorflow.keras` framework for building deep learning models.
- `utils` is a module that contains helper functions for this assignment. You do not need to modify the code in this file.

Run the cell below to import all the necessary packages.


```python
import time
from collections import deque, namedtuple

import gym
import numpy as np
import PIL.Image
import tensorflow as tf
import utils

from pyvirtualdisplay import Display
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam
```


```python
# Set up a virtual display to render the Lunar Lander environment.
Display(visible=0, size=(840, 480)).start();

# Set the random seed for TensorFlow
tf.random.set_seed(utils.SEED)
```

<a name="2"></a>
## 2 - Hyperparameters

Run the cell below to set the hyperparameters.


```python
MEMORY_SIZE = 100_000     # size of memory buffer
GAMMA = 0.995             # discount factor
ALPHA = 1e-3              # learning rate  
NUM_STEPS_FOR_UPDATE = 4  # perform a learning update every C time steps
```

<a name="3"></a>
## 3 - The Lunar Lander Environment

In this notebook we will be using [OpenAI's Gym Library](https://www.gymlibrary.dev/). The Gym library provides a wide variety of environments for reinforcement learning. To put it simply, an environment represents a problem or task to be solved. In this notebook, we will try to solve the Lunar Lander environment using reinforcement learning.

The goal of the Lunar Lander environment is to land the lunar lander safely on the landing pad on the surface of the moon. The landing pad is designated by two flag poles and its center is at coordinates `(0,0)` but the lander is also allowed to land outside of the landing pad. The lander starts at the top center of the environment with a random initial force applied to its center of mass and has infinite fuel. The environment is considered solved if you get `200` points. 

<br>
<br>
<figure>
  <img src = "images/lunar_lander.gif" width = 40%>
      <figcaption style = "text-align: center; font-style: italic">Fig 1. Lunar Lander Environment.</figcaption>
</figure>



<a name="3.1"></a>
### 3.1 Action Space

The agent has four discrete actions available:

* Do nothing.
* Fire right engine.
* Fire main engine.
* Fire left engine.

Each action has a corresponding numerical value:

```python
Do nothing = 0
Fire right engine = 1
Fire main engine = 2
Fire left engine = 3
```

<a name="3.2"></a>
### 3.2 Observation Space

The agent's observation space consists of a state vector with 8 variables:

* Its $(x,y)$ coordinates. The landing pad is always at coordinates $(0,0)$.
* Its linear velocities $(\dot x,\dot y)$.
* Its angle $\theta$.
* Its angular velocity $\dot \theta$.
* Two booleans, $l$ and $r$, that represent whether each leg is in contact with the ground or not.

<a name="3.3"></a>
### 3.3 Rewards

After every step, a reward is granted. The total reward of an episode is the sum of the rewards for all the steps within that episode.

For each step, the reward:
- is increased/decreased the closer/further the lander is to the landing pad.
- is increased/decreased the slower/faster the lander is moving.
- is decreased the more the lander is tilted (angle not horizontal).
- is increased by 10 points for each leg that is in contact with the ground.
- is decreased by 0.03 points each frame a side engine is firing.
- is decreased by 0.3 points each frame the main engine is firing.

The episode receives an additional reward of -100 or +100 points for crashing or landing safely respectively.

<a name="3.4"></a>
### 3.4 Episode Termination

An episode ends (i.e the environment enters a terminal state) if:

* The lunar lander crashes (i.e if the body of the lunar lander comes in contact with the surface of the moon).

* The absolute value of the lander's $x$-coordinate is greater than 1 (i.e. it goes beyond the left or right border)

You can check out the [Open AI Gym documentation](https://www.gymlibrary.dev/environments/box2d/lunar_lander/) for a full description of the environment. 

<a name="4"></a>
## 4 - Load the Environment

We start by loading the `LunarLander-v2` environment from the `gym` library by using the `.make()` method. `LunarLander-v2` is the latest version of the Lunar Lander environment and you can read about its version history in the [Open AI Gym documentation](https://www.gymlibrary.dev/environments/box2d/lunar_lander/#version-history).


```python
env = gym.make('LunarLander-v2')
```

Once we load the environment we use the `.reset()` method to reset the environment to the initial state. The lander starts at the top center of the environment and we can render the first frame of the environment by using the `.render()` method.


```python
env.reset()
PIL.Image.fromarray(env.render(mode='rgb_array'))
```




![png](output_11_0.png)



In order to build our neural network later on we need to know the size of the state vector and the number of valid actions. We can get this information from our environment by using the `.observation_space.shape` and `action_space.n` methods, respectively.


```python
state_size = env.observation_space.shape
num_actions = env.action_space.n

print('State Shape:', state_size)
print('Number of actions:', num_actions)
```

    State Shape: (8,)
    Number of actions: 4


<a name="5"></a>
## 5 - Interacting with the Gym Environment

The Gym library implements the standard “agent-environment loop” formalism:

<br>
<center>
<video src = "./videos/rl_formalism.m4v" width="840" height="480" controls autoplay loop poster="./images/rl_formalism.png"> </video>
<figcaption style = "text-align:center; font-style:italic">Fig 2. Agent-environment Loop Formalism.</figcaption>
</center>
<br>

In the standard “agent-environment loop” formalism, an agent interacts with the environment in discrete time steps $t=0,1,2,...$. At each time step $t$, the agent uses a policy $\pi$ to select an action $A_t$ based on its observation of the environment's state $S_t$. The agent receives a numerical reward $R_t$ and on the next time step, moves to a new state $S_{t+1}$.

<a name="5.1"></a>
### 5.1 Exploring the Environment's Dynamics

In Open AI's Gym environments, we use the `.step()` method to run a single time step of the environment's dynamics. In the version of `gym` that we are using the `.step()` method accepts an action and returns four values:

* `observation` (**object**): an environment-specific object representing your observation of the environment. In the Lunar Lander environment this corresponds to a numpy array containing the positions and velocities of the lander as described in section [3.2 Observation Space](#3.2).


* `reward` (**float**): amount of reward returned as a result of taking the given action. In the Lunar Lander environment this corresponds to a float of type `numpy.float64` as described in section [3.3 Rewards](#3.3).


* `done` (**boolean**): When done is `True`, it indicates the episode has terminated and it’s time to reset the environment. 


* `info` (**dictionary**): diagnostic information useful for debugging. We won't be using this variable in this notebook but it is shown here for completeness.

To begin an episode, we need to reset the environment to an initial state. We do this by using the `.reset()` method. 


```python
# Reset the environment and get the initial state.
current_state = env.reset()
```

Once the environment is reset, the agent can start taking actions in the environment by using the `.step()` method. Note that the agent can only take one action per time step. 

In the cell below you can select different actions and see how the returned values change depending on the action taken. Remember that in this environment the agent has four discrete actions available and we specify them in code by using their corresponding numerical value:

```python
Do nothing = 0
Fire right engine = 1
Fire main engine = 2
Fire left engine = 3
```


```python
# Select an action
action = 0

# Run a single time step of the environment's dynamics with the given action.
next_state, reward, done, _ = env.step(action)

# Display table with values.
utils.display_table(current_state, action, next_state, reward, done)

# Replace the `current_state` with the state after the action is taken
current_state = next_state
```


<style  type="text/css" >
    #T_4d3b86c4_a8f1_11ef_bd30_0242ac120041 th {
          border: 1px solid grey;
          text-align: center;
    }    #T_4d3b86c4_a8f1_11ef_bd30_0242ac120041 tbody td {
          border: 1px solid grey;
          text-align: center;
    }    #T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row0_col0 {
            background-color :  grey;
        }    #T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row1_col1 {
            background-color :  grey;
        }    #T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row1_col2 {
            background-color :  grey;
        }    #T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row1_col3 {
            background-color :  grey;
        }    #T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row1_col4 {
            background-color :  grey;
        }    #T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row1_col5 {
            background-color :  grey;
        }    #T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row1_col6 {
            background-color :  grey;
        }    #T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row1_col7 {
            background-color :  grey;
        }    #T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row1_col8 {
            background-color :  grey;
        }    #T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row1_col9 {
            background-color :  grey;
        }    #T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row1_col10 {
            background-color :  grey;
        }    #T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row1_col11 {
            background-color :  grey;
        }    #T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row2_col0 {
            background-color :  grey;
        }    #T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row3_col1 {
            background-color :  grey;
        }    #T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row3_col2 {
            background-color :  grey;
        }    #T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row3_col3 {
            background-color :  grey;
        }    #T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row3_col4 {
            background-color :  grey;
        }    #T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row3_col5 {
            background-color :  grey;
        }    #T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row3_col6 {
            background-color :  grey;
        }    #T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row3_col7 {
            background-color :  grey;
        }    #T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row3_col8 {
            background-color :  grey;
        }    #T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row3_col9 {
            background-color :  grey;
        }    #T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row3_col10 {
            background-color :  grey;
        }    #T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row3_col11 {
            background-color :  grey;
        }    #T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row4_col1 {
            background-color :  grey;
        }    #T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row4_col2 {
            background-color :  grey;
        }    #T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row4_col3 {
            background-color :  grey;
        }    #T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row4_col4 {
            background-color :  grey;
        }    #T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row4_col5 {
            background-color :  grey;
        }    #T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row4_col6 {
            background-color :  grey;
        }    #T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row4_col7 {
            background-color :  grey;
        }    #T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row4_col8 {
            background-color :  grey;
        }    #T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row4_col9 {
            background-color :  grey;
        }    #T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row4_col10 {
            background-color :  grey;
        }    #T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row4_col11 {
            background-color :  grey;
        }</style><table id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" ></th>        <th class="col_heading level0 col1" colspan=8>State Vector</th>        <th class="col_heading level0 col9" colspan=3>Derived from the State Vector (the closer to zero, the better)</th>    </tr>    <tr>        <th class="blank level1" ></th>        <th class="col_heading level1 col0" ></th>        <th class="col_heading level1 col1" colspan=2>Coordinate</th>        <th class="col_heading level1 col3" colspan=2>Velocity</th>        <th class="col_heading level1 col5" colspan=2>Tilting</th>        <th class="col_heading level1 col7" colspan=2>Ground contact</th>        <th class="col_heading level1 col9" >Distance from landing pad</th>        <th class="col_heading level1 col10" >Velocity</th>        <th class="col_heading level1 col11" >Tilting Angle (absolute value)</th>    </tr>    <tr>        <th class="blank level2" ></th>        <th class="col_heading level2 col0" ></th>        <th class="col_heading level2 col1" >X (Horizontal)</th>        <th class="col_heading level2 col2" >Y (Vertical)</th>        <th class="col_heading level2 col3" >X (Horizontal)</th>        <th class="col_heading level2 col4" >Y (Vertical)</th>        <th class="col_heading level2 col5" >Angle</th>        <th class="col_heading level2 col6" >Angular Velocity</th>        <th class="col_heading level2 col7" >Left Leg?</th>        <th class="col_heading level2 col8" >Right Leg?</th>        <th class="col_heading level2 col9" ></th>        <th class="col_heading level2 col10" ></th>        <th class="col_heading level2 col11" ></th>    </tr></thead><tbody>
                <tr>
                        <th id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041level0_row0" class="row_heading level0 row0" >Current State</th>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row0_col0" class="data row0 col0" ></td>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row0_col1" class="data row0 col1" >0.001919</td>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row0_col2" class="data row0 col2" >1.422301</td>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row0_col3" class="data row0 col3" >0.194400</td>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row0_col4" class="data row0 col4" >0.505814</td>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row0_col5" class="data row0 col5" >-0.002217</td>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row0_col6" class="data row0 col6" >-0.044034</td>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row0_col7" class="data row0 col7" >False</td>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row0_col8" class="data row0 col8" >False</td>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row0_col9" class="data row0 col9" >1.422302</td>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row0_col10" class="data row0 col10" >0.541885</td>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row0_col11" class="data row0 col11" >0.002217</td>
            </tr>
            <tr>
                        <th id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041level0_row1" class="row_heading level0 row1" >Action</th>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row1_col0" class="data row1 col0" >Do nothing</td>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row1_col1" class="data row1 col1" ></td>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row1_col2" class="data row1 col2" ></td>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row1_col3" class="data row1 col3" ></td>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row1_col4" class="data row1 col4" ></td>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row1_col5" class="data row1 col5" ></td>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row1_col6" class="data row1 col6" ></td>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row1_col7" class="data row1 col7" ></td>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row1_col8" class="data row1 col8" ></td>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row1_col9" class="data row1 col9" ></td>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row1_col10" class="data row1 col10" ></td>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row1_col11" class="data row1 col11" ></td>
            </tr>
            <tr>
                        <th id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041level0_row2" class="row_heading level0 row2" >Next State</th>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row2_col0" class="data row2 col0" ></td>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row2_col1" class="data row2 col1" >0.003839</td>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row2_col2" class="data row2 col2" >1.433103</td>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row2_col3" class="data row2 col3" >0.194137</td>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row2_col4" class="data row2 col4" >0.480094</td>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row2_col5" class="data row2 col5" >-0.004393</td>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row2_col6" class="data row2 col6" >-0.043519</td>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row2_col7" class="data row2 col7" >False</td>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row2_col8" class="data row2 col8" >False</td>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row2_col9" class="data row2 col9" >1.433108</td>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row2_col10" class="data row2 col10" >0.517860</td>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row2_col11" class="data row2 col11" >0.004393</td>
            </tr>
            <tr>
                        <th id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041level0_row3" class="row_heading level0 row3" >Reward</th>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row3_col0" class="data row3 col0" >1.104326</td>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row3_col1" class="data row3 col1" ></td>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row3_col2" class="data row3 col2" ></td>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row3_col3" class="data row3 col3" ></td>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row3_col4" class="data row3 col4" ></td>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row3_col5" class="data row3 col5" ></td>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row3_col6" class="data row3 col6" ></td>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row3_col7" class="data row3 col7" ></td>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row3_col8" class="data row3 col8" ></td>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row3_col9" class="data row3 col9" ></td>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row3_col10" class="data row3 col10" ></td>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row3_col11" class="data row3 col11" ></td>
            </tr>
            <tr>
                        <th id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041level0_row4" class="row_heading level0 row4" >Episode Terminated</th>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row4_col0" class="data row4 col0" >False</td>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row4_col1" class="data row4 col1" ></td>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row4_col2" class="data row4 col2" ></td>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row4_col3" class="data row4 col3" ></td>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row4_col4" class="data row4 col4" ></td>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row4_col5" class="data row4 col5" ></td>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row4_col6" class="data row4 col6" ></td>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row4_col7" class="data row4 col7" ></td>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row4_col8" class="data row4 col8" ></td>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row4_col9" class="data row4 col9" ></td>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row4_col10" class="data row4 col10" ></td>
                        <td id="T_4d3b86c4_a8f1_11ef_bd30_0242ac120041row4_col11" class="data row4 col11" ></td>
            </tr>
    </tbody></table>


In practice, when we train the agent we use a loop to allow the agent to take many consecutive actions during an episode.

<a name="6"></a>
## 6 - Deep Q-Learning

In cases where both the state and action space are discrete we can estimate the action-value function iteratively by using the Bellman equation:

$$
Q_{i+1}(s,a) = R + \gamma \max_{a'}Q_i(s',a')
$$

This iterative method converges to the optimal action-value function $Q^*(s,a)$ as $i\to\infty$. This means that the agent just needs to gradually explore the state-action space and keep updating the estimate of $Q(s,a)$ until it converges to the optimal action-value function $Q^*(s,a)$. However, in cases where the state space is continuous it becomes practically impossible to explore the entire state-action space. Consequently, this also makes it practically impossible to gradually estimate $Q(s,a)$ until it converges to $Q^*(s,a)$.

In the Deep $Q$-Learning, we solve this problem by using a neural network to estimate the action-value function $Q(s,a)\approx Q^*(s,a)$. We call this neural network a $Q$-Network and it can be trained by adjusting its weights at each iteration to minimize the mean-squared error in the Bellman equation.

Unfortunately, using neural networks in reinforcement learning to estimate action-value functions has proven to be highly unstable. Luckily, there's a couple of techniques that can be employed to avoid instabilities. These techniques consist of using a ***Target Network*** and ***Experience Replay***. We will explore these two techniques in the following sections.

<a name="6.1"></a>
### 6.1 Target Network

We can train the $Q$-Network by adjusting it's weights at each iteration to minimize the mean-squared error in the Bellman equation, where the target values are given by:

$$
y = R + \gamma \max_{a'}Q(s',a';w)
$$

where $w$ are the weights of the $Q$-Network. This means that we are adjusting the weights $w$ at each iteration to minimize the following error:

$$
\overbrace{\underbrace{R + \gamma \max_{a'}Q(s',a'; w)}_{\rm {y~target}} - Q(s,a;w)}^{\rm {Error}}
$$

Notice that this forms a problem because the $y$ target is changing on every iteration. Having a constantly moving target can lead to oscillations and instabilities. To avoid this, we can create
a separate neural network for generating the $y$ targets. We call this separate neural network the **target $\hat Q$-Network** and it will have the same architecture as the original $Q$-Network. By using the target $\hat Q$-Network, the above error becomes:

$$
\overbrace{\underbrace{R + \gamma \max_{a'}\hat{Q}(s',a'; w^-)}_{\rm {y~target}} - Q(s,a;w)}^{\rm {Error}}
$$

where $w^-$ and $w$ are the weights of the target $\hat Q$-Network and $Q$-Network, respectively.

In practice, we will use the following algorithm: every $C$ time steps we will use the $\hat Q$-Network to generate the $y$ targets and update the weights of the target $\hat Q$-Network using the weights of the $Q$-Network. We will update the weights $w^-$ of the the target $\hat Q$-Network using a **soft update**. This means that we will update the weights $w^-$ using the following rule:
 
$$
w^-\leftarrow \tau w + (1 - \tau) w^-
$$

where $\tau\ll 1$. By using the soft update, we are ensuring that the target values, $y$, change slowly, which greatly improves the stability of our learning algorithm.

<a name="ex01"></a>
### Exercise 1

In this exercise you will create the $Q$ and target $\hat Q$ networks and set the optimizer. Remember that the Deep $Q$-Network (DQN) is a neural network that approximates the action-value function $Q(s,a)\approx Q^*(s,a)$. It does this by learning how to map states to $Q$ values.

To solve the Lunar Lander environment, we are going to employ a DQN with the following architecture:

* An `Input` layer that takes `state_size` as input.

* A `Dense` layer with `64` units and a `relu` activation function.

* A `Dense` layer with `64` units and a `relu` activation function.

* A `Dense` layer with `num_actions` units and a `linear` activation function. This will be the output layer of our network.


In the cell below you should create the $Q$-Network and the target $\hat Q$-Network using the model architecture described above. Remember that both the $Q$-Network and the target $\hat Q$-Network have the same architecture.

Lastly, you should set `Adam` as the optimizer with a learning rate equal to `ALPHA`. Recall that `ALPHA` was defined in the [Hyperparameters](#2) section. We should note that for this exercise you should use the already imported packages:
```python
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
```


```python
# UNQ_C1
# GRADED CELL

# Create the Q-Network
q_network = Sequential([
    ### START CODE HERE ### 
    Input(shape=state_size),                      
    Dense(units=64, activation='relu'),            
    Dense(units=64, activation='relu'),            
    Dense(units=num_actions, activation='linear'),
    ### END CODE HERE ### 
    ])

# Create the target Q^-Network
target_q_network = Sequential([
    ### START CODE HERE ### 
    Input(shape=state_size),                       
    Dense(units=64, activation='relu'),            
    Dense(units=64, activation='relu'),            
    Dense(units=num_actions, activation='linear'), 
    ### END CODE HERE ###
    ])

### START CODE HERE ### 
optimizer = Adam(learning_rate=ALPHA)
### END CODE HERE ###
```


```python
# UNIT TEST
from public_tests import *

test_network(q_network)
test_network(target_q_network)
test_optimizer(optimizer, ALPHA) 
```

    [92mAll tests passed!
    [92mAll tests passed!
    [92mAll tests passed!


<details>
  <summary><font size="3" color="darkgreen"><b>Click for hints</b></font></summary>
    
```python
# Create the Q-Network
q_network = Sequential([
    Input(shape=state_size),                      
    Dense(units=64, activation='relu'),            
    Dense(units=64, activation='relu'),            
    Dense(units=num_actions, activation='linear'),
    ])

# Create the target Q^-Network
target_q_network = Sequential([
    Input(shape=state_size),                       
    Dense(units=64, activation='relu'),            
    Dense(units=64, activation='relu'),            
    Dense(units=num_actions, activation='linear'), 
    ])

optimizer = Adam(learning_rate=ALPHA)                                  
``` 

<a name="6.2"></a>
### 6.2 Experience Replay

When an agent interacts with the environment, the states, actions, and rewards the agent experiences are sequential by nature. If the agent tries to learn from these consecutive experiences it can run into problems due to the strong correlations between them. To avoid this, we employ a technique known as **Experience Replay** to generate uncorrelated experiences for training our agent. Experience replay consists of storing the agent's experiences (i.e the states, actions, and rewards the agent receives) in a memory buffer and then sampling a random mini-batch of experiences from the buffer to do the learning. The experience tuples $(S_t, A_t, R_t, S_{t+1})$ will be added to the memory buffer at each time step as the agent interacts with the environment.

For convenience, we will store the experiences as named tuples.


```python
# Store experiences as named tuples
experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
```

By using experience replay we avoid problematic correlations, oscillations and instabilities. In addition, experience replay also allows the agent to potentially use the same experience in multiple weight updates, which increases data efficiency.

<a name="7"></a>
## 7 - Deep Q-Learning Algorithm with Experience Replay

Now that we know all the techniques that we are going to use, we can put them together to arrive at the Deep Q-Learning Algorithm With Experience Replay.
<br>
<br>
<figure>
  <img src = "images/deep_q_algorithm.png" width = 90% style = "border: thin silver solid; padding: 0px">
      <figcaption style = "text-align: center; font-style: italic">Fig 3. Deep Q-Learning with Experience Replay.</figcaption>
</figure>

<a name="ex02"></a>
### Exercise 2

In this exercise you will implement line ***12*** of the algorithm outlined in *Fig 3* above and you will also compute the loss between the $y$ targets and the $Q(s,a)$ values. In the cell below, complete the `compute_loss` function by setting the $y$ targets equal to:

$$
\begin{equation}
    y_j =
    \begin{cases}
      R_j & \text{if episode terminates at step  } j+1\\
      R_j + \gamma \max_{a'}\hat{Q}(s_{j+1},a') & \text{otherwise}\\
    \end{cases}       
\end{equation}
$$

Here are a couple of things to note:

* The `compute_loss` function takes in a mini-batch of experience tuples. This mini-batch of experience tuples is unpacked to extract the `states`, `actions`, `rewards`, `next_states`, and `done_vals`. You should keep in mind that these variables are *TensorFlow Tensors* whose size will depend on the mini-batch size. For example, if the mini-batch size is `64` then both `rewards` and `done_vals` will be TensorFlow Tensors with `64` elements.


* Using `if/else` statements to set the $y$ targets will not work when the variables are tensors with many elements. However, notice that you can use the `done_vals` to implement the above in a single line of code. To do this, recall that the `done` variable is a Boolean variable that takes the value `True` when an episode terminates at step $j+1$ and it is `False` otherwise. Taking into account that a Boolean value of `True` has the numerical value of `1` and a Boolean value of `False` has the numerical value of `0`, you can use the factor `(1 - done_vals)` to implement the above in a single line of code. Here's a hint: notice that `(1 - done_vals)` has a value of `0` when `done_vals` is `True` and a value of `1` when `done_vals` is `False`. 

Lastly, compute the loss by calculating the Mean-Squared Error (`MSE`) between the `y_targets` and the `q_values`. To calculate the mean-squared error you should use the already imported package `MSE`:
```python
from tensorflow.keras.losses import MSE
```


```python
# UNQ_C2
# GRADED FUNCTION: calculate_loss

def compute_loss(experiences, gamma, q_network, target_q_network):
    """ 
    Calculates the loss.
    
    Args:
      experiences: (tuple) tuple of ["state", "action", "reward", "next_state", "done"] namedtuples
      gamma: (float) The discount factor.
      q_network: (tf.keras.Sequential) Keras model for predicting the q_values
      target_q_network: (tf.keras.Sequential) Keras model for predicting the targets
          
    Returns:
      loss: (TensorFlow Tensor(shape=(0,), dtype=int32)) the Mean-Squared Error between
            the y targets and the Q(s,a) values.
    """

    # Unpack the mini-batch of experience tuples
    states, actions, rewards, next_states, done_vals = experiences
    
    # Compute max Q^(s,a)
    max_qsa = tf.reduce_max(target_q_network(next_states), axis=-1)
    
    # Set y = R if episode terminates, otherwise set y = R + γ max Q^(s,a).
    ### START CODE HERE ### 
    y_targets = rewards + (gamma * max_qsa * (1 - done_vals))
    ### END CODE HERE ###
    
    # Get the q_values and reshape to match y_targets
    q_values = q_network(states)
    q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]),
                                                tf.cast(actions, tf.int32)], axis=1))
        
    # Compute the loss
    ### START CODE HERE ### 
    loss = MSE(y_targets, q_values) 
    ### END CODE HERE ### 
    
    return loss
```


```python
# UNIT TEST    
test_compute_loss(compute_loss)
```

    [92mAll tests passed!


<details>
  <summary><font size="3" color="darkgreen"><b>Click for hints</b></font></summary>
    
```python
def compute_loss(experiences, gamma, q_network, target_q_network):
    """ 
    Calculates the loss.
    
    Args:
      experiences: (tuple) tuple of ["state", "action", "reward", "next_state", "done"] namedtuples
      gamma: (float) The discount factor.
      q_network: (tf.keras.Sequential) Keras model for predicting the q_values
      target_q_network: (tf.keras.Sequential) Keras model for predicting the targets
          
    Returns:
      loss: (TensorFlow Tensor(shape=(0,), dtype=int32)) the Mean-Squared Error between
            the y targets and the Q(s,a) values.
    """

    
    # Unpack the mini-batch of experience tuples
    states, actions, rewards, next_states, done_vals = experiences
    
    # Compute max Q^(s,a)
    max_qsa = tf.reduce_max(target_q_network(next_states), axis=-1)
    
    # Set y = R if episode terminates, otherwise set y = R + γ max Q^(s,a).
    y_targets = rewards + (gamma * max_qsa * (1 - done_vals))
    
    # Get the q_values
    q_values = q_network(states)
    q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]),
                                                tf.cast(actions, tf.int32)], axis=1))
    
    # Calculate the loss
    loss = MSE(y_targets, q_values)
    
    return loss

``` 
    

<a name="8"></a>
## 8 - Update the Network Weights

We will use the `agent_learn` function below to implement lines ***12 -14*** of the algorithm outlined in [Fig 3](#7). The `agent_learn` function will update the weights of the $Q$ and target $\hat Q$ networks using a custom training loop. Because we are using a custom training loop we need to retrieve the gradients via a `tf.GradientTape` instance, and then call `optimizer.apply_gradients()` to update the weights of our $Q$-Network. Note that we are also using the `@tf.function` decorator to increase performance. Without this decorator our training will take twice as long. If you would like to know more about how to increase performance with `@tf.function` take a look at the [TensorFlow documentation](https://www.tensorflow.org/guide/function).

The last line of this function updates the weights of the target $\hat Q$-Network using a [soft update](#6.1). If you want to know how this is implemented in code we encourage you to take a look at the `utils.update_target_network` function in the `utils` module.


```python
@tf.function
def agent_learn(experiences, gamma):
    """
    Updates the weights of the Q networks.
    
    Args:
      experiences: (tuple) tuple of ["state", "action", "reward", "next_state", "done"] namedtuples
      gamma: (float) The discount factor.
    
    """
    
    # Calculate the loss
    with tf.GradientTape() as tape:
        loss = compute_loss(experiences, gamma, q_network, target_q_network)

    # Get the gradients of the loss with respect to the weights.
    gradients = tape.gradient(loss, q_network.trainable_variables)
    
    # Update the weights of the q_network.
    optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))

    # update the weights of target q_network
    utils.update_target_network(q_network, target_q_network)
```

<a name="9"></a>
## 9 - Train the Agent

We are now ready to train our agent to solve the Lunar Lander environment. In the cell below we will implement the algorithm in [Fig 3](#7) line by line (please note that we have included the same algorithm below for easy reference. This will prevent you from scrolling up and down the notebook):

* **Line 1**: We initialize the `memory_buffer` with a capacity of $N =$ `MEMORY_SIZE`. Notice that we are using a `deque` as the data structure for our `memory_buffer`.


* **Line 2**: We skip this line since we already initialized the `q_network` in [Exercise 1](#ex01).


* **Line 3**: We initialize the `target_q_network` by setting its weights to be equal to those of the `q_network`.


* **Line 4**: We start the outer loop. Notice that we have set $M =$ `num_episodes = 2000`. This number is reasonable because the agent should be able to solve the Lunar Lander environment in less than `2000` episodes using this notebook's default parameters.


* **Line 5**: We use the `.reset()` method to reset the environment to the initial state and get the initial state.


* **Line 6**: We start the inner loop. Notice that we have set $T =$ `max_num_timesteps = 1000`. This means that the episode will automatically terminate if the episode hasn't terminated after `1000` time steps.


* **Line 7**: The agent observes the current `state` and chooses an `action` using an $\epsilon$-greedy policy. Our agent starts out using a value of $\epsilon =$ `epsilon = 1` which yields an $\epsilon$-greedy policy that is equivalent to the equiprobable random policy. This means that at the beginning of our training, the agent is just going to take random actions regardless of the observed `state`. As training progresses we will decrease the value of $\epsilon$ slowly towards a minimum value using a given $\epsilon$-decay rate. We want this minimum value to be close to zero because a value of $\epsilon = 0$ will yield an $\epsilon$-greedy policy that is equivalent to the greedy policy. This means that towards the end of training, the agent will lean towards selecting the `action` that it believes (based on its past experiences) will maximize $Q(s,a)$. We will set the minimum $\epsilon$ value to be `0.01` and not exactly 0 because we always want to keep a little bit of exploration during training. If you want to know how this is implemented in code we encourage you to take a look at the `utils.get_action` function in the `utils` module.


* **Line 8**: We use the `.step()` method to take the given `action` in the environment and get the `reward` and the `next_state`. 


* **Line 9**: We store the `experience(state, action, reward, next_state, done)` tuple in our `memory_buffer`. Notice that we also store the `done` variable so that we can keep track of when an episode terminates. This allowed us to set the $y$ targets in [Exercise 2](#ex02).


* **Line 10**: We check if the conditions are met to perform a learning update. We do this by using our custom `utils.check_update_conditions` function. This function checks if $C =$ `NUM_STEPS_FOR_UPDATE = 4` time steps have occured and if our `memory_buffer` has enough experience tuples to fill a mini-batch. For example, if the mini-batch size is `64`, then our `memory_buffer` should have more than `64` experience tuples in order to pass the latter condition. If the conditions are met, then the `utils.check_update_conditions` function will return a value of `True`, otherwise it will return a value of `False`.


* **Lines 11 - 14**: If the `update` variable is `True` then we perform a learning update. The learning update consists of sampling a random mini-batch of experience tuples from our `memory_buffer`, setting the $y$ targets, performing gradient descent, and updating the weights of the networks. We will use the `agent_learn` function we defined in [Section 8](#8) to perform the latter 3.


* **Line 15**: At the end of each iteration of the inner loop we set `next_state` as our new `state` so that the loop can start again from this new state. In addition, we check if the episode has reached a terminal state (i.e we check if `done = True`). If a terminal state has been reached, then we break out of the inner loop.


* **Line 16**: At the end of each iteration of the outer loop we update the value of $\epsilon$, and check if the environment has been solved. We consider that the environment has been solved if the agent receives an average of `200` points in the last `100` episodes. If the environment has not been solved we continue the outer loop and start a new episode.

Finally, we wanted to note that we have included some extra variables to keep track of the total number of points the agent received in each episode. This will help us determine if the agent has solved the environment and it will also allow us to see how our agent performed during training. We also use the `time` module to measure how long the training takes. 

<br>
<br>
<figure>
  <img src = "images/deep_q_algorithm.png" width = 90% style = "border: thin silver solid; padding: 0px">
      <figcaption style = "text-align: center; font-style: italic">Fig 4. Deep Q-Learning with Experience Replay.</figcaption>
</figure>
<br>

**Note:** With this notebook's default parameters, the following cell takes between 10 to 15 minutes to run. 


```python
start = time.time()

num_episodes = 2000
max_num_timesteps = 1000

total_point_history = []

num_p_av = 100    # number of total points to use for averaging
epsilon = 1.0     # initial ε value for ε-greedy policy

# Create a memory buffer D with capacity N
memory_buffer = deque(maxlen=MEMORY_SIZE)

# Set the target network weights equal to the Q-Network weights
target_q_network.set_weights(q_network.get_weights())

for i in range(num_episodes):
    
    # Reset the environment to the initial state and get the initial state
    state = env.reset()
    total_points = 0
    
    for t in range(max_num_timesteps):
        
        # From the current state S choose an action A using an ε-greedy policy
        state_qn = np.expand_dims(state, axis=0)  # state needs to be the right shape for the q_network
        q_values = q_network(state_qn)
        action = utils.get_action(q_values, epsilon)
        
        # Take action A and receive reward R and the next state S'
        next_state, reward, done, _ = env.step(action)
        
        # Store experience tuple (S,A,R,S') in the memory buffer.
        # We store the done variable as well for convenience.
        memory_buffer.append(experience(state, action, reward, next_state, done))
        
        # Only update the network every NUM_STEPS_FOR_UPDATE time steps.
        update = utils.check_update_conditions(t, NUM_STEPS_FOR_UPDATE, memory_buffer)
        
        if update:
            # Sample random mini-batch of experience tuples (S,A,R,S') from D
            experiences = utils.get_experiences(memory_buffer)
            
            # Set the y targets, perform a gradient descent step,
            # and update the network weights.
            agent_learn(experiences, GAMMA)
        
        state = next_state.copy()
        total_points += reward
        
        if done:
            break
            
    total_point_history.append(total_points)
    av_latest_points = np.mean(total_point_history[-num_p_av:])
    
    # Update the ε value
    epsilon = utils.get_new_eps(epsilon)

    print(f"\rEpisode {i+1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}", end="")

    if (i+1) % num_p_av == 0:
        print(f"\rEpisode {i+1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}")

    # We will consider that the environment is solved if we get an
    # average of 200 points in the last 100 episodes.
    if av_latest_points >= 200.0:
        print(f"\n\nEnvironment solved in {i+1} episodes!")
        q_network.save('lunar_lander_model.h5')
        break
        
tot_time = time.time() - start

print(f"\nTotal Runtime: {tot_time:.2f} s ({(tot_time/60):.2f} min)")
```

    Episode 100 | Total point average of the last 100 episodes: -150.85
    Episode 200 | Total point average of the last 100 episodes: -106.11
    Episode 300 | Total point average of the last 100 episodes: -77.256
    Episode 400 | Total point average of the last 100 episodes: -25.01
    Episode 500 | Total point average of the last 100 episodes: 159.91
    Episode 534 | Total point average of the last 100 episodes: 201.37
    
    Environment solved in 534 episodes!
    
    Total Runtime: 741.21 s (12.35 min)


We can plot the total point history along with the moving average to see how our agent improved during training. If you want to know about the different plotting options available in the `utils.plot_history` function we encourage you to take a look at the `utils` module.


```python
# Plot the total point history along with the moving average
utils.plot_history(total_point_history)
```


![png](output_38_0.png)


<a name="10"></a>
## 10 - See the Trained Agent In Action

Now that we have trained our agent, we can see it in action. We will use the `utils.create_video` function to create a video of our agent interacting with the environment using the trained $Q$-Network. The `utils.create_video` function uses the `imageio` library to create the video. This library produces some warnings that can be distracting, so, to suppress these warnings we run the code below.


```python
# Suppress warnings from imageio
import logging
logging.getLogger().setLevel(logging.ERROR)
```

In the cell below we create a video of our agent interacting with the Lunar Lander environment using the trained `q_network`. The video is saved to the `videos` folder with the given `filename`. We use the `utils.embed_mp4` function to embed the video in the Jupyter Notebook so that we can see it here directly without having to download it.

We should note that since the lunar lander starts with a random initial force applied to its center of mass, every time you run the cell below you will see a different video. If the agent was trained properly, it should be able to land the lunar lander in the landing pad every time, regardless of the initial force applied to its center of mass.


```python
filename = "./videos/lunar_lander.mp4"

utils.create_video(filename, env, q_network)
utils.embed_mp4(filename)
```





<video width="840" height="480" controls>
<source src="data:video/mp4;base64,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AQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAgAAAAAAgAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAGAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAIAAAAAAIAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAIAAAAAAIAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAABAAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAIAAAAAAIAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAYAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAYAAAAAAQAAAgAAAAABAAAEAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACAAAAAACAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACAAAAAACAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACAAAAAACAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAoAAAAAAQAABAAAAAABAAAAAAAAAAEAAAIAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAQAAAAAAQAACgAAAAABAAAEAAAAAAEAAAAAAAAAAQAAAgAAAAABAAAKAAAAAAEAAAQAAAAAAQAAAAAAAAABAAACAAAAAAEAAAgAAAAAAgAAAgAAAAAcc3RzYwAAAAAAAAABAAAAAQAAAgAAAAABAAAIFHN0c3oAAAAAAAAAAAAAAgAAAAjDAAABEwAAAFYAAAA/AAAASAAAAIUAAAA6AAAAOwAAADQAAABrAAAAKgAAAB4AAAAXAAAAUAAAACMAAAAbAAAAIQAAACAAAAAkAAAAGwAAABsAAADCAAAAOAAAACsAAAAtAAAAkwAAAD8AAAAvAAAAPQAAALEAAABSAAAAQAAAAE0AAAEAAAAAQgAAAE0AAACwAAAA3QAAAFAAAABHAAAAWwAAAWUAAABZAAABnwAAAMQAAACyAAAAnwAAAUUAAADNAAAAwwAAALQAAAEMAAAA2QAAAMYAAAC8AAABWAAAAMoAAACzAAAAmwAAAWAAAACLAAAAfwAAAIsAAAFKAAAAsgAAAIkAAACwAAABVwAAAMkAAACqAAAApAAAAW4AAACwAAAAtQAAANQAAAFqAAAAnAAAAKkAAAC8AAABJwAAANAAAADBAAAAqgAAAPMAAACrAAAA3AAAALwAAADBAAAAnQAAAJ8AAAB2AAAA4wAAAEwAAABAAAAAxgAAAFEAAAA5AAAATAAAAKUAAAB1AAAAcwAAAF4AAACtAAAAVAAAAFkAAABBAAAApAAAAFcAAAFVAAAAtQAAAGkAAACaAAABFQAAANkAAADfAAAA0wAAAW0AAADoAAAAzAAAAMQAAAEaAAAAwQAAAJoAAACqAAAA6QAAANEAAAC2AAABTAAAARUAAACvAAAA1wAAAP4AAADNAAAAxwAAAKMAAAFeAAAA8QAAAM0AAADOAAABPQAAANAAAAC/AAAA4AAAAREAAACXAAAApAAAALoAAAFWAAAAyQAAAIMAAADDAAABTgAAAPkAAADqAAAAxAAAAWAAAADrAAAAxgAAANIAAAEhAAABDAAAALIAAACVAAABLwAAANcAAACrAAAAuAAAAWUAAACkAAAAnQAAAJ0AAAFmAAAAtQAAAIkAAACKAAAA5AAAAMsAAADAAAAAxAAAARAAAACyAAAAgAAAAJYAAAFjAAAAwAAAAGkAAACBAAABAAAAAKgAAACwAAAAmgAAAN4AAACKAAAApQAAAUEAAACrAAAAugAAAIkAAAE5AAAAswAAAJ0AAACuAAABMwAAANkAAADXAAAA1AAAAPEAAADnAAAAlwAAAIYAAAB6AAABAgAAAIkAAABgAAAAVAAAAPIAAADKAAAAfAAAAIMAAAEHAAAAjQAAAJEAAABvAAABkAAAAMQAAACSAAAArAAAAT8AAAC9AAAAwwAAANIAAAERAAAArAAAAKQAAADxAAAAvgAAAMEAAACdAAABDQAAAJwAAACOAAAAkQAAAOMAAACIAAABAQAAAMMAAACIAAAAjQAAANgAAAC4AAAI8AAAAT8AAADKAAAAigAAAKgAAADlAAAAoAAAAIoAAACKAAABFgAAAKgAAAB/AAAAhwAAARMAAADcAAAAlwAAAIcAAAD3AAAAmAAAAJwAAAB4AAAAvgAAAHcAAABvAAAAdgAAAOEAAABuAAAAXQAAAFoAAAEpAAAAswAAAJcAAACNAAAA0gAAAJIAAABrAAAAcAAAAQUAAABjAAAAWQAAAHoAAAEHAAAAgAAAAGcAAADHAAAAZAAAAFAAAABhAAAAfQAAAF0AAABBAAABAQAAAGgAAABVAAAAdAAAAPEAAABWAAAAcgAAAJEAAACwAAAAXAAAAGMAAAA8AAAAzgAAAGUAAABAAAAAZwAAAPwAAAB1AAAAhQAAAH4AAADqAAAAjQAAAE4AAABqAAAAwwAAAFoAAABlAAAAXQAAARQAAACwAAAAawAAAG8AAAFxAAAAzwAAAHoAAACMAAABMgAAAKIAAACtAAAAowAAAS4AAADOAAAAsAAAAH0AAAEHAAAAwgAAAHoAAABzAAAAxwAAAG4AAABYAAAAaAAAANoAAABrAAAASAAAAHgAAAC9AAAAaQAAAG0AAAB2AAAA9gAAAG0AAABnAAAAagAAANAAAACPAAAAQQAAAHoAAADiAAAAigAAAGUAAAB4AAAA2wAAAHQAAABJAAAASgAAAOAAAAB8AAAAbAAAAFsAAACNAAAAdwAAAHYAAABHAAAA4wAAAGEAAABAAAAAbAAAAQsAAABpAAAAVgAAAQsAAAB0AAAAhgAAAHkAAADrAAAArAAAAGEAAACNAAAA5gAAAIgAAABiAAAAXwAAAJkAAAB5AAAAYgAAAFEAAADXAAAAlQAAAHoAAABpAAAA3wAAAI0AAAB5AAAAbgAAAPMAAAB2AAAATgAAAFMAAAC4AAAASwAAAEUAAABXAAABHwAAAHYAAABMAAAAbAAAANwAAACVAAAAdgAAAHsAAACuAAAAZQAAAFcAAAA8AAAA2wAAAGIAAABWAAAAUwAAAJUAAABkAAAAUgAAAEcAAADFAAAAhAAAAFAAAABeAAAAugAAAF8AAABcAAAAVQAAANAAAABXAAAARgAAAEkAAAESAAAAmgAAAFoAAABnAAAAxgAAAGAAAABmAAAAVgAAARkAAABoAAAAKQAAAGIAAAEWAAAAjAAAAFcAAABaAAAAuwAAADkAAAA0AAAAMgAAAPgAAAA3AAAAKQAAAB8AAACBAAAAKwAAAB8AAAAeAAAAhAAAAC4AAAAkAAAAHQAAAB4AAAAoAAAAHAAAABsAAAAcAAAAJwAAABsAAAAbAAAAHAAAACcAAAAbAAAAGwAABvMAAAAlAAAAGgAAABcAAAAUAAAAGwAAABgAAAAUAAAAFAAAABsAAAAYAAAAFAAAABRzdGNvAAAAAAAAAAEAAAAwAAAAYnVkdGEAAABabWV0YQAAAAAAAAAhaGRscgAAAAAAAAAAbWRpcmFwcGwAAAAAAAAAAAAAAAAtaWxzdAAAACWpdG9vAAAAHWRhdGEAAAABAAAAAExhdmY1OC4yOS4xMDA=" type="video/mp4">
Your browser does not support the video tag.
</video>



<a name="11"></a>
## 11 - Congratulations!

You have successfully used Deep Q-Learning with Experience Replay to train an agent to land a lunar lander safely on a landing pad on the surface of the moon. Congratulations!

<a name="12"></a>
## 12 - References

If you would like to learn more about Deep Q-Learning, we recommend you check out the following papers.


* Mnih, V., Kavukcuoglu, K., Silver, D. et al. Human-level control through deep reinforcement learning. Nature 518, 529–533 (2015).


* Lillicrap, T. P., Hunt, J. J., Pritzel, A., et al. Continuous Control with Deep Reinforcement Learning. ICLR (2016).


* Mnih, V., Kavukcuoglu, K., Silver, D. et al. Playing Atari with Deep Reinforcement Learning. arXiv e-prints.  arXiv:1312.5602 (2013).

<details>
  <summary><font size="2" color="darkgreen"><b>Please click here if you want to experiment with any of the non-graded code.</b></font></summary>
    <p><i><b>Important Note: Please only do this when you've already passed the assignment to avoid problems with the autograder.</b></i>
    <ol>
        <li> On the notebook’s menu, click “View” > “Cell Toolbar” > “Edit Metadata”</li>
        <li> Hit the “Edit Metadata” button next to the code cell which you want to lock/unlock</li>
        <li> Set the attribute value for “editable” to:
            <ul>
                <li> “true” if you want to unlock it </li>
                <li> “false” if you want to lock it </li>
            </ul>
        </li>
        <li> On the notebook’s menu, click “View” > “Cell Toolbar” > “None” </li>
    </ol>
    <p> Here's a short demo of how to do the steps above: 
        <br>
        <img src="https://lh3.google.com/u/0/d/14Xy_Mb17CZVgzVAgq7NCjMVBvSae3xO1" align="center" alt="unlock_cells.gif">
</details>
