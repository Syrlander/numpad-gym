from argparse import ArgumentError
import gym
from gym import spaces
import numpy as np
from rl_thesis.environments.numpad_base import Numpad
from .config import Config


class Numpad2DDiscrete(Numpad):
    metadata = {"render.modes": ["human"]}

    NUM_ACTIONS = 4
    NUM_CHANNELS = 2

    LIGHTS_CHANNEL, BALL_CHANNEL = 0, 1

    NO_BALL, BALL = 0, 1
    LIGHT_OFF, LIGHT_ON = 0, 1

    # Actions
    UP, RIGHT, DOWN, LEFT = (0, 1, 2, 3)

    def __init__(self, config: Config):
        super(Numpad2DDiscrete, self).__init__(config)

        self.first_render = True

        self.flatten_state = config.flatten_state
        self.action_space = spaces.Discrete(self.NUM_ACTIONS)
        if not self.flatten_state:
            self.observation_space = spaces.Box(
                low=0,
                high=1,
                shape=(self.NUM_CHANNELS, config.numpad_size,
                       config.numpad_size),
                dtype=np.uint8,
            )
        else:
            self.observation_space = spaces.Box(
                low=0,
                high=1,
                shape=(self.NUM_CHANNELS * config.numpad_size *
                       config.numpad_size, ),
                dtype=np.uint8,
            )

        # Initialize state - possibly overwriting ball start position and sequence
        self.reset()

        if config.ball_start and config.seq:
            self.state[self.BALL_CHANNEL, self.ball_pos[0],
                       self.ball_pos[1]] = self.NO_BALL
            self.state[self.BALL_CHANNEL, config.ball_start[0],
                       config.ball_start[1]] = self.BALL
            self.ball_pos = config.ball_start

            self.seq = config.seq
            self.seq_len = len(config.seq)

    def __seq_num_to_idx(self, seq_num):
        return int(seq_num / self.numpad_size), seq_num % self.numpad_size

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(
                f"Got invalid action: {action}\nAction space confined to: {self.action_space}"
            )

        info = {}
        reward = 0

        if self.done:
            return self.state, reward, self.done, info

        self.t += 1
        self.done = self.t == self.t_max

        # Clear current ball position
        bi, bj = self.ball_pos
        self.state[self.BALL_CHANNEL, bi, bj] = self.NO_BALL

        # Move ball to next position
        if action == self.UP:
            bi -= 1
        elif action == self.RIGHT:
            bj += 1
        elif action == self.DOWN:
            bi += 1
        elif action == self.LEFT:
            bj -= 1
        else:
            raise ArgumentError(
                f"Got invalid action: {action}. Action space is: {self.action_space}"
            )

        self.ball_pos = np.clip(bi, 0, self.numpad_size - 1), np.clip(
            bj, 0, self.numpad_size - 1)
        self.state[self.BALL_CHANNEL, self.ball_pos[0],
                   self.ball_pos[1]] = self.BALL

        #print(f"took action: {action} - ball step onto: {self.ball_pos}")
        #self.stepped_on.add((self.ball_pos[0], self.ball_pos[1]))

        seq_next = self.__seq_num_to_idx(self.seq[self.seq_cnt])
        if not self.ball_pos == seq_next:
            self.state[self.LIGHTS_CHANNEL] = self.LIGHT_OFF
            self.seq_cnt = 0

        # Check if next ball position is the next sequence position
        seq_next = self.__seq_num_to_idx(self.seq[self.seq_cnt])
        if self.ball_pos == seq_next:
            self.state[self.LIGHTS_CHANNEL, self.ball_pos[0],
                       self.ball_pos[1]] = self.LIGHT_ON

            if self.seq_cnt > self.seq_max_cnt:
                reward = 1
                self.seq_max_cnt = self.seq_cnt

            self.seq_cnt += 1

            # Completing sequence before time is up:
            # Reset lights and sequence counters, so the agent can
            # obtain more reward by using the 'memorized' sequence
            if self.seq_max_cnt == self.seq_len - 1:
                # return state where all lights are on, so the
                # reward and the observation match as in the other cases
                # but internally use a state where all lights are off, so
                # it is ready for the model to try and run sequence again
                tmp_state = self.state.copy()
                if self.flatten_state:
                    tmp_state = tmp_state.flatten()
                self.state[self.LIGHTS_CHANNEL] = self.LIGHT_OFF

                self.seq_cnt = 0
                self.seq_max_cnt = -1
                return tmp_state, reward, self.done, info

        #if self.done:
        #    print(f"Stepped on tiles: {self.stepped_on}")
        to_return = []
        if self.flatten_state:
            to_return.append(self.state.flatten())
        else:
            to_return.append(self.state)
        to_return.extend([reward, self.done, info])
        return to_return

    def reset(self):
        """
        Reset episode - create new sequence
        """
        # Initialize state - channels:
        #   0: lights channel - indicates which lights are on
        #   1: ball channel - location of ball
        self.state = np.tile(
            self.LIGHT_OFF,
            (self.NUM_CHANNELS, self.numpad_size, self.numpad_size))
        self.t = 0
        self.done = False

        # Initialize sequence
        self.seq_cnt = 0
        self.seq_max_cnt = -1  # Track the max. revealed sequence cnt
        self.seq = self.generate_sequence(self.numpad_size, self.seq_len)
        seq_start = self.__seq_num_to_idx(self.seq[0])

        # Initialize ball position at random - cannot start on first tile of sequence
        ball_start = self.rng.randint(self.numpad_size, size=2)
        while (ball_start != seq_start).all():
            ball_start = self.rng.randint(self.numpad_size, size=2)

        self.state[self.BALL_CHANNEL, ball_start[0], ball_start[1]] = self.BALL
        self.ball_pos = ball_start

        #self.stepped_on = set()
        #print(f"seq: {[self.__seq_num_to_idx(s) for s in self.seq]}")
        #print(f"Ball starting at: {self.ball_pos}")
        #self.stepped_on.add((self.ball_pos[0], self.ball_pos[1]))

        # First observation = initial state
        if self.flatten_state:
            return self.state.flatten()
        else:
            return self.state

    def render(self, mode="human"):
        if self.first_render:
            print(f"Seq.: {self.seq}")
            self.first_render = False

        print("Tiles:")
        print(self.state[self.LIGHTS_CHANNEL])
        print("Ball:")
        print(self.state[self.BALL_CHANNEL])
        print()

    def close(self):
        pass