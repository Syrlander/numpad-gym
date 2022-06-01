from abc import ABC
import gym 
import numpy as np
from .config import Config

class Numpad(gym.Env, ABC):
    def __init__(self, config : Config):
        super(Numpad, self).__init__()
        self.numpad_size = config.numpad_size
        self.seq_len = config.seq_len
        self.t_max = config.t_max
        self.ball_start = config.ball_start
        self.seq = config.seq
        self.rng = np.random.RandomState(seed=config.seed)
       
    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode="file", img_filepath="numpad_continuous", skip_frames=5, remove_frames=True):
        pass

    def close (self):
        pass
    

    def neighbor_tiles(self, numpad_size, tile):
        """
            Directions in comments assumes numpad with structure as
            0 1 2
            3 4 5
            6 7 8
        """
        neighbors = []
        n_tiles = numpad_size**2
        
        # checks that tile is not on the right edge of the numpad, so that going right would "wrap around"
        if not tile % numpad_size == numpad_size - 1:
            # neighbor on the right 
            neighbors.append(tile + 1)
        
        # checks that tile is not on the left edge of the numpad
        if not tile % numpad_size == 0:
            # neighbor on the left 
            neighbors.append(tile - 1)
            
        # neighbor below
        neighbors.append(tile + numpad_size)
        # neighbor above
        neighbors.append(tile - numpad_size)
        
        # remove all out of bounds tiles
        return list(filter(lambda x: x >= 0 and x < n_tiles, neighbors))


    def generate_sequence(self, numpad_size, sequence_length):
        seq = []
        n_tiles = numpad_size**2
        start_tile = self.rng.randint(0, n_tiles)
        seq.append(start_tile)
        prev_tile = start_tile
        # subtract one because we already found one tile
        for i in range(sequence_length - 1):
            options = self.neighbor_tiles(numpad_size, prev_tile)
            # remove already taken tiles
            options = list(filter(lambda x: x not in seq, options))
            if len(options) == 0:
                return self.generate_sequence(numpad_size, sequence_length)
            next_tile = self.rng.choice(options)
            seq.append(next_tile)
            prev_tile = next_tile
        return seq