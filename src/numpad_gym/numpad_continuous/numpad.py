# from argparse import ArgumentError
from argparse import ArgumentError
from multiprocessing.sharedctypes import Value
import gym
from gym import spaces
import numpy as np
from PIL import Image
from pathlib import Path
import shutil

from torch import Argument
from rl_thesis.environments.numpad_base import Numpad
from .config import Config
import warnings


class Numpad2DContinuous(Numpad):
    metadata = {
        "render.modes": ["array", "print", "file", "gif"],
        "observation.modes": ["rgb_array", "greyscale_array", "ram"]
    }

    # RGB color definitions for rgb and greyscale array observation modes
    GROUND = np.array([192, 192, 192], dtype=np.uint8)
    BALL = np.array([255, 0, 0], dtype=np.uint8)
    LIGHT_ON = np.array([255, 255, 102], dtype=np.uint8)
    LIGHT_OFF = np.array([50, 50, 0], dtype=np.uint8)

    ACTION_LOW, ACTION_HIGH = -1, 1

    DIVISION_BY_ZERO_PREVENT = 1e-10

    def __init__(self, config: Config):
        super(Numpad2DContinuous, self).__init__(config)

        # We first consider the environment as initialized upon the first call to reset.
        # Used to ensure that reset is called before any calls to step
        self.initialized = False

        self.task_cues = config.task_cues
        self.task_cue_threshold = config.task_cue_threshold

        self.tile_size = config.tile_size
        self.ball_size = config.ball_size
        if self.ball_size % 2 == 0:
            warnings.warn(
                f"Note ball size is even (ball_size = {self.ball_size}), hence it's center is not well defined. It's recommended to use an odd ball size."
            )
        self.numpad_size = config.numpad_size
        self.num_tiles = self.numpad_size**2

        # Actions in [-1, 1], multiplied by some normalization factor depending on image size
        self.action_space = spaces.Box(low=self.ACTION_LOW,
                                       high=self.ACTION_HIGH,
                                       shape=(2, ),
                                       dtype=float)

        # For more on these see computation of next ball position under step()
        self.acc_scalar = self.tile_size / 10
        self.max_velocity = self.tile_size / 2
        self.max_velocity_norm = np.linalg.norm(
            np.array([self.max_velocity, self.max_velocity]))

        # Height and width of generated square image
        self.img_size = config.numpad_size * config.tile_size + (
            2 * config.numpad_size
        ) * config.spacing_size  # pixels of all tiles + pixels of all spacing

        # Observation space from 0-1 due to normalization
        self.observation_mode = config.observation_mode

        if self.observation_mode == "rgb_array":
            self.observation_space = spaces.Box(low=0,
                                                high=1,
                                                shape=(3, self.img_size,
                                                       self.img_size),
                                                dtype=float)
        elif self.observation_mode == "greyscale_array":
            self.observation_space = spaces.Box(low=0,
                                                high=1,
                                                shape=(1, self.img_size,
                                                       self.img_size),
                                                dtype=float)
        elif self.observation_mode == "ram":
            # Normalized to (-1)-1 due to ball velocity
            # tiles/lights are either 0 (off) or 1 (on)

            self.observation_space = spaces.Box(
                low=-1,
                high=1,
                shape=(
                    (4 + self.num_tiles, )
                ),  # ball pos vec. (2 dim.) + ball velocity (2 dim.) + ball speed + num. tiles 
                dtype=float)
        else:
            raise ValueError(
                "Got invalid NumPad2D continuous mode.\nAvailable modes are:\n\t'rgb_array': observation space is a 3 color channel 2D array of the board.\n\t'greyscale_array': observation space is a single channel 2D array of the board.\n\t'ram': observation space consists of ball position and binary values indicating if lights in sequence are on or off."
            )

        # Both ball_start and a sequence must be given, not just either one
        # If they are specified they are used through all episodes
        if bool(config.ball_start) ^ bool(config.seq):
            raise ValueError(
                f"Either ball_start or seq was given while the other wasn't. Both must be specified or neither of them.\nGot values: ball_start: {config.ball_start}, seq: {config.seq}"
            )

        self.seq_len = config.seq_len
        self.use_specified_init_values = False

        if config.ball_start and config.seq:
            self.init_ball_pos = np.array(config.ball_start, dtype=float)
            self.init_seq = config.seq
            self.seq_len = len(self.init_seq)
            self.use_specified_init_values = True

    def __seq_num_to_idx(self, seq_num):
        return np.array(
            [int(seq_num / self.numpad_size), seq_num % self.numpad_size])

    def __idx_to_pos(self, idx):
        # first term finds the upper left corner of the tile (including spacing), second term moves it to the center of the tile
        return idx * (self.img_size / self.numpad_size) + (
            self.img_size / self.numpad_size) / 2

    def __seq_num_to_pos(self, seq_num):
        idx = self.__seq_num_to_idx(seq_num)
        pos = self.__idx_to_pos(idx)
        return pos

    def __pos_to_seq_num(self, pos):
        """
        Args:
            pos: position on the board

        Returns:
            Sequence number of tile if the given position is on a tile.
            None, otherwise.
        """
        # find the tile disregarding spacing between tiles
        idx = (pos / (self.img_size / self.numpad_size)).astype(int)
        center_of_tile = self.__idx_to_pos(idx)
        tile_bounds_left, tile_bounds_right = center_of_tile - self.tile_size / 2, center_of_tile + self.tile_size / 2

        # check if ball is on tile, and return sequence number if true, else return None
        if pos[0] > tile_bounds_left[0] and pos[1] > tile_bounds_left[
                1] and pos[0] < tile_bounds_right[0] and pos[
                    1] < tile_bounds_right[1]:
            seq_num = idx[0] * self.numpad_size + idx[1]
            return seq_num
        else:
            return None

    def __draw_square(self, rgb_img, center, size, color):
        rounded_pos = np.round(center)
        start, end = rounded_pos - int(size / 2), rounded_pos + int(size / 2)
        start = np.clip(start, 0, self.img_size - 1).astype(int)
        end = np.clip(end, 0, self.img_size - 1).astype(int)
        rgb_img[:, start[0]:end[0] + 1,
                start[1]:end[1] + 1] = color.reshape(-1, 1, 1)

    def __draw_ball(self, rgb_img, pos):
        self.__draw_square(rgb_img, pos, self.ball_size, self.BALL)

    def __turn_on_light(self, rgb_img, seq_num):
        pos = self.__seq_num_to_pos(seq_num)
        self.__draw_square(rgb_img, pos, self.tile_size, self.LIGHT_ON)

    def __turn_off_light(self, rgb_img, seq_num):
        pos = self.__seq_num_to_pos(seq_num)
        self.__draw_square(rgb_img, pos, self.tile_size, self.LIGHT_OFF)

    def __set_lights(self, rgb_img, lights):
        for i, light_status in enumerate(lights):
            if light_status == 0:
                self.__turn_off_light(rgb_img, i)
            elif light_status == 1:
                self.__turn_on_light(rgb_img, i)
            else:
                raise ValueError(f"Got invalid lights configuration: {lights}")

    def __generate_observation(self, copy=False):
        if self.observation_mode == "rgb_array" or self.observation_mode == "greyscale_array":
            return self.render(mode="array")
        elif self.observation_mode == "ram":
            if copy:
                lights_obs = self.lights.copy()
            else:
                lights_obs = self.lights

            velocity_u = self.ball_velocity / self.max_velocity
            return np.concatenate(
                [self.ball_pos / self.img_size, velocity_u, lights_obs])

    def step(self, action):
        if not self.initialized:
            raise RuntimeError(
                "Attempted to take an action before initializing environment. Please ensure you run reset() before any calls to step(action)"
            )

        if not self.action_space.contains(action):
            raise ValueError(
                f"Got invalid action: {action}. Action space is: {self.action_space}"
            )

        if self.t == 0 and self.task_cues:
            # Reset lights on first step if using task cues, as reset() will set
            # all lights as being on
            self.lights[self.seq] = 0

        info = {}
        reward = 0

        if self.done:
            # If taking an action even if episodes is done, return final observation
            observation = np.concatenate(
                [self.ball_pos / self.img_size, self.lights])
            return observation, reward, self.done, info

        # Accelerate with 1/10 of tile size and cap. max. velocity in any direction to half of tile size, to avoid issues of moving over tiles
        self.ball_velocity += np.clip(action * self.acc_scalar,
                                      -self.max_velocity, self.max_velocity)
        self.ball_pos += self.ball_velocity
        self.ball_pos = np.clip(self.ball_pos, 0, self.img_size - 1)

        # if ball hits a wall set velocity in the wall's direction to 0
        if self.ball_pos[0] == 0 or self.ball_pos[0] == self.img_size - 1:
            self.ball_velocity[0] = 0
        if self.ball_pos[1] == 0 or self.ball_pos[1] == self.img_size - 1:
            self.ball_velocity[1] = 0

        self.t += 1
        self.done = self.t == self.t_max

        curr_seq_num = self.__pos_to_seq_num(
            self.ball_pos)  # == None if on spacing
        if curr_seq_num != None and curr_seq_num != self.prev_seq_num:
            if curr_seq_num == self.seq[self.next_seq_ind]:
                # Standing on next tile of sequence

                if self.next_seq_ind > self.max_seq_ind:
                    # Standing on previous unseen tile
                    reward = 1
                    self.max_seq_ind = self.next_seq_ind

                self.lights[self.seq[self.next_seq_ind]] = 1

                self.next_seq_ind += 1

                if self.next_seq_ind == self.seq_len:
                    # Standing on last tile of sequence
                    self.max_seq_ind = -1
                    self.next_seq_ind = 0

                    observation = self.__generate_observation(copy=True)

                    self.lights = np.zeros(self.num_tiles)

                    return observation, reward, self.done, info
            else:
                # Not standing on next tile
                self.lights = np.zeros(self.num_tiles)
                self.next_seq_ind = 0

        self.prev_seq_num = curr_seq_num

        observation = self.__generate_observation()
        return observation, reward, self.done, info

    def reset(self):
        self.t = 0
        self.done = False

        if self.use_specified_init_values:
            self.seq = self.init_seq
        else:
            self.seq = self.generate_sequence(self.numpad_size, self.seq_len)
        self.max_seq_ind = -1  # max. index seen of the hidden sequence so far
        self.next_seq_ind = 0  # next sequence index of tile to light up

        self.prev_seq_num = None  # Track last sequence number ball was on

        self.ball_velocity = np.array([0, 0], dtype=float)

        # Ensure ball doesn't start on first tile of sequence
        if self.use_specified_init_values:
            self.ball_pos = self.init_ball_pos.copy()
        else:
            self.ball_pos = self.rng.random(2) * self.img_size
            while self.__pos_to_seq_num(self.ball_pos) == self.seq[0]:
                self.ball_pos = self.rng.random(2) * self.img_size

        # All lights off by default
        self.lights = np.zeros(self.num_tiles, dtype=float)

        if self.task_cues:
            # NOTE: Humplik et al. have "partial task cues during training" (page 20)
            # where the task cues uses a random mask, to determine which lights to show.
            task_cue_mask = self.rng.random(self.seq_len) >= self.task_cue_threshold
            print(f"task_cue_mask: {task_cue_mask}")
            print(f"seq: {self.seq}")
            self.lights[np.array(self.seq)[task_cue_mask]] = 1
            print(f"lights: {self.lights}")

        self.initialized = True

        obs = self.__generate_observation()

        return obs

    FILE_RENDER_MODES = {"rgb_array", "greyscale_array"}

    def render(
        self,
        mode="file",
        img_filepath="numpad_continuous",
        frames_freq=5,
        remove_frame_files=True,
        normalize=True,
    ):
        """
        Render the environment in a specified format

        Kwargs:
            mode: string declaring the render format
            img_filepath: filepath of where to save image when using "file" render mode
            frames_freq: frequency of how often to save a frame/image of the environment if using "gif" render mode
            remove_frame_files: whether or not to remove the individual frames (images) generated for the construction of the gif in "gif" render mode

        Remarks:
            Only certain combinations of 
        """
        if not self.initialized:
            raise RuntimeError(
                "Attempted to render before initializing environment. Please ensure you run reset() before any calls to render()"
            )

        if mode == "array":
            if self.observation_mode == "rgb_array":
                rgb_img = np.stack([
                    np.tile(self.GROUND[0], (self.img_size, self.img_size)),
                    np.tile(self.GROUND[1], (self.img_size, self.img_size)),
                    np.tile(self.GROUND[2], (self.img_size, self.img_size)),
                ])

                self.__set_lights(rgb_img, self.lights)
                self.__draw_ball(rgb_img, self.ball_pos)

                if normalize:
                    rgb_img = rgb_img.astype(float)
                    rgb_img /= 255

                return rgb_img
            elif self.observation_mode == "greyscale_array":
                self.observation_mode = "rgb_array"
                norm_rgb_img = self.render(mode="array", normalize=normalize)
                self.observation_mode = "greyscale_array"

                return np.mean(norm_rgb_img, axis=0, keepdims=True)
            elif self.observation_mode == "ram":
                return np.concatenate(
                    [self.ball_pos / self.img_size, self.lights])
        elif mode == "print":
            print(f"observation mode: {self.observation_mode}")
            print("#" * 10)
            print(self.render(mode="array"))
        elif mode == "file":
            if self.observation_mode not in self.FILE_RENDER_MODES:
                raise ValueError(
                    f"'file' render mode not available for '{self.observation_mode}' observation mode.\nOnly observation modes {self.FILE_RENDER_MODES} can be used with 'file' render mode."
                )

            arr = self.render(file="array", normalize=False)

            img_mode = "RGB"
            if self.observation_mode == "greyscale_array":
                arr_ra = arr_ra.squeeze().astype(np.uint8)
                img_mode = "L"

            im = Image.fromarray(np.rollaxis(arr.transpose((0, 2, 1)), 0, 3),
                                 mode=img_mode)
            im.save(f"{img_filepath}.png")
        elif mode == "gif":
            # if self.observation_mode not in self.FILE_RENDER_MODES:
            #     raise ValueError(f"'file' render mode not available for '{self.observation_mode}' observation mode.\nOnly observation modes {self.FILE_RENDER_MODES} can be used with 'file' render mode.")

            rem = self.t_max % frames_freq
            if rem != 0:
                raise ValueError(
                    f"t_max must be a multiple of frames_freq.\nGot t_max: {self.t_max}, frames_freq: {frames_freq}, remainder: {rem}"
                )

            # Create temp frame dir
            frames_dir = Path(".frames")
            frames_dir.mkdir(exist_ok=True)

            # Save an image every frames_freq
            if self.t % frames_freq == 0:
                tmp_observation_mode = self.observation_mode
                self.observation_mode = "rgb_array"
                arr = self.render(mode="array", normalize=False)
                self.observation_mode = tmp_observation_mode
                # print(f"array shape: {arr.shape}")
                arr_t = arr.transpose((0, 2, 1))
                # print(f"array_t shape: {arr_t.shape}")
                arr_ra = np.rollaxis(arr_t, 0, 3)
                # print(f"array_ra shape: {arr_ra.shape}")

                img_mode = "RGB"
                if self.observation_mode == "greyscale_array":
                    arr_ra = arr_ra.squeeze().astype(np.uint8)
                    img_mode = "L"

                im = Image.fromarray(arr_ra, mode=img_mode)
                im.save(Path(frames_dir, f"{self.t}.png"))

            # Construct gif on final frame - remove all frame images
            if (self.t + 1) == self.t_max:
                gif_path = Path(f"{img_filepath}.gif")

                # Playback in sorted order by frame number
                img, *imgs = [
                    Image.open(file)
                    for file in sorted(frames_dir.glob("*.png"),
                                       key=lambda f: int(f.name.split(".")[0]))
                ]
                img.save(fp=gif_path,
                         format="GIF",
                         append_images=imgs,
                         save_all=True,
                         duration=200,
                         loop=0)

                if remove_frame_files:
                    shutil.rmtree(frames_dir)

                # Attempt to optimize gif using: https://imageio.readthedocs.io/en/stable/examples.html#optimizing-a-gif-using-pygifsicle
                try:
                    from pygifsicle import optimize
                    optimize(str(gif_path), options=["--no-warnings"])
                except (FileNotFoundError, ImportError):
                    print(
                        "gifsicle not available - unable to optimize gif size")
        else:
            modes_str = ''.join([
                f'\n\t{available_mode}'
                for available_mode in self.metadata['render.modes']
            ])
            raise ValueError(
                f"Got invalid render mode: '{mode}'.\nAvailable modes:{modes_str}"
            )

    def close(self):
        pass
