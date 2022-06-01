import pytest

import numpy as np

from numpad_gym.numpad_discrete.numpad import Numpad2DDiscrete
from numpad_gym.numpad_discrete.config import Config as DiscreteConfig

from numpad_gym.numpad_continuous.numpad import Numpad2DContinuous
from numpad_gym.numpad_continuous.config import Config as ContinuousConfig


class TestNumpad2DDiscrete:

    def setup_method(self):
        conf = DiscreteConfig()
        conf.ball_start = (1, 1)
        conf.seq = [0, 3, 4, 5]
        conf.t_max = 50

        self.env = Numpad2DDiscrete(conf)

    def test_initial_state(self):
        light_state, ball_state = self.env.state
        assert ball_state.sum() == 1
        assert ball_state[1, 1] == 1
        assert (light_state == 0).all()

    def test_up(self):
        self.env.step(self.env.UP)
        light_state, ball_state = self.env.state
        assert ball_state.sum() == 1
        assert ball_state[0, 1] == 1
        assert (light_state == 0).all()

    def test_down(self):
        self.env.step(self.env.DOWN)
        light_state, ball_state = self.env.state
        assert ball_state.sum() == 1
        assert ball_state[2, 1] == 1
        assert (light_state == 0).all()

    def test_left(self):
        self.env.step(self.env.LEFT)
        light_state, ball_state = self.env.state
        assert ball_state.sum() == 1
        assert ball_state[1, 0] == 1
        assert (light_state == 0).all()

    def test_right(self):
        self.env.step(self.env.RIGHT)
        light_state, ball_state = self.env.state
        assert ball_state.sum() == 1
        assert ball_state[1, 2] == 1
        assert (light_state == 0).all()

    def test_move_against_bound(self):
        self.env.step(self.env.UP)
        self.env.step(self.env.UP)
        self.env.step(self.env.UP)
        self.env.step(self.env.UP)
        self.env.step(self.env.UP)
        assert self.env.state[self.env.BALL_CHANNEL, 0, 1] == 1

        self.env.step(self.env.LEFT)
        self.env.step(self.env.LEFT)
        self.env.step(self.env.LEFT)
        self.env.step(self.env.LEFT)
        self.env.step(self.env.LEFT)
        assert self.env.state[self.env.BALL_CHANNEL, 0, 0] == 1

        self.env.step(self.env.RIGHT)
        self.env.step(self.env.RIGHT)
        self.env.step(self.env.RIGHT)
        self.env.step(self.env.RIGHT)
        self.env.step(self.env.RIGHT)
        assert self.env.state[self.env.BALL_CHANNEL, 0, 2] == 1

        self.env.step(self.env.DOWN)
        self.env.step(self.env.DOWN)
        self.env.step(self.env.DOWN)
        self.env.step(self.env.DOWN)
        self.env.step(self.env.DOWN)
        assert self.env.state[self.env.BALL_CHANNEL, 2, 2] == 1

    def test_reset(self):
        np.random.seed(12345)

        env = Numpad2DDiscrete(DiscreteConfig())
        ball_pos1, seq1 = env.ball_pos, env.seq
        env.reset()
        ball_pos2, seq2 = env.ball_pos, env.seq

        assert (ball_pos1 != ball_pos2).any()
        assert seq1 != seq2

    def test_multiple_rewards(self):
        _, rew, _, _ = self.env.step(self.env.UP)
        assert rew == 0
        _, rew, _, _ = self.env.step(self.env.LEFT)
        assert rew == 1
        _, rew, _, _ = self.env.step(self.env.DOWN)
        assert rew == 1

        # There should be no reward on stepping back to first in sequence
        _, rew, _, _ = self.env.step(self.env.UP)
        assert rew == 0

    def test_reset_lights(self):
        (light_state, _), _, _, _ = self.env.step(self.env.UP)
        assert (light_state == 0).all()

        (light_state, _), _, _, _ = self.env.step(self.env.LEFT)
        assert light_state[0, 0] == 1
        assert light_state.sum() == 1

        (light_state, _), _, _, _ = self.env.step(self.env.DOWN)
        assert light_state[0, 0] == 1
        assert light_state[1, 0] == 1
        assert light_state.sum() == 2

        (light_state, _), _, _, _ = self.env.step(self.env.UP)
        assert light_state[0, 0] == 1
        assert light_state.sum() == 1

        (light_state, _), _, _, _ = self.env.step(self.env.RIGHT)
        assert (light_state == 0).all()

    def test_reset_rewards_before_timeup(self):
        # Test that we can still receive more reward after having
        # completed the entire sequence
        _, rew, _, _ = self.env.step(self.env.UP)
        assert rew == 0
        _, rew, _, _ = self.env.step(self.env.LEFT)
        assert rew == 1
        _, rew, _, _ = self.env.step(self.env.DOWN)
        assert rew == 1
        _, rew, _, _ = self.env.step(self.env.RIGHT)
        assert rew == 1
        _, rew, _, _ = self.env.step(self.env.RIGHT)
        assert rew == 1

        # Sequence is now done try to run again
        _, rew, _, _ = self.env.step(self.env.UP)
        assert rew == 0
        _, rew, _, _ = self.env.step(self.env.LEFT)
        assert rew == 0
        _, rew, _, _ = self.env.step(self.env.LEFT)
        assert rew == 1
        _, rew, _, _ = self.env.step(self.env.DOWN)
        assert rew == 1
        _, rew, _, _ = self.env.step(self.env.RIGHT)
        assert rew == 1

    def test_move_from_last_to_first(self):
        # Ensure that the observation of moving from the last position in the
        # sequence to the first, after having completed the sequence, only
        # contains a single light being on.
        conf = DiscreteConfig()
        conf.ball_start = (1, 0)
        conf.seq_len = 2
        conf.seq = [0, 1]
        conf.t_max = 50

        env = Numpad2DDiscrete(conf)

        (light_state, _), _, _, _ = env.step(env.UP)
        assert light_state[0, 0] == 1
        assert light_state.sum() == 1

        (light_state, _), _, _, _ = env.step(env.RIGHT)
        assert light_state[0, 0] == 1
        assert light_state[0, 1] == 1
        assert light_state.sum() == 2

        (light_state, _), _, _, _ = env.step(env.LEFT)
        assert light_state[0, 0] == 1
        assert light_state.sum() == 1

    def test_timeup_done(self):
        done = False

        for _ in range(50):
            _, _, done, _ = self.env.step(self.env.UP)

        assert done == True

    def test_move_out_of_bounds_next_seq(self):
        # Test that if the ball is at a bound and it's current position cause a
        # tile to light up in the previous move and the ball moves against the
        # bound. Then the light should turn off, as we wish to penalize moving
        # the ball against the bound
        (light_state, _), rew, _, _ = self.env.step(self.env.UP)
        assert light_state.sum() == 0
        assert rew == 0

        (light_state, _), rew, _, _ = self.env.step(self.env.LEFT)
        assert light_state.sum() == 1
        assert rew == 1

        # Note it is correct that if the ball 'bumps' into the wall on the first
        # tile, then the ball lands back on the first tile, so the first tile
        # is still lit up.
        (light_state, _), rew, _, _ = self.env.step(self.env.LEFT)
        assert light_state.sum() == 1
        assert rew == 0

        (light_state, _), rew, _, _ = self.env.step(self.env.DOWN)
        assert light_state.sum() == 2
        assert rew == 1

        (light_state, _), rew, _, _ = self.env.step(self.env.LEFT)
        assert light_state.sum() == 0
        assert rew == 0

    def test_env_seed(self):
        """
        Ensure runs across different environment instances are the same when a 
        seed is provided in the environment config.
        """
        conf = DiscreteConfig()
        conf.seed = 1234

        env1 = Numpad2DDiscrete(conf)
        env2 = Numpad2DDiscrete(conf)

        for _ in range(10):
            env1.reset()
            env2.reset()

            assert (env1.ball_pos == env2.ball_pos).all()
            assert env1.seq == env2.seq

            for _ in range(conf.t_max):
                action = env1.action_space.sample()

                obs1, rew1, done1, info1 = env1.step(action)
                obs2, rew2, done2, info2 = env2.step(action)

                assert (obs1 == obs2).all()
                assert rew1 == rew2
                assert done1 == done2
                assert info1 == info2

    def test_flatten_observations(self):
        conf = DiscreteConfig()
        conf.flatten_state = True
        conf.numpad_size = 3

        env = Numpad2DDiscrete(conf)

        done = False
        obs = env.reset()
        assert len(obs) == 18

        while not done:
            obs, _, done, _ = env.step(env.UP)
            assert len(obs) == 18

class TestNumpad2DContinuous:
    """
    3x3 example env. has sequence numbers:
    0 3 6
    1 4 7
    2 5 8
    """

    def init_3x3(self,
                 observation_mode="ram",
                 spacing_size=100,
                 tile_size=50,
                 ball_size=5,
                 t_max=50,
                 ball_start=[0, 0],
                 seq=[6, 7, 8],
                 task_cues=False,
                 seed=None):
        conf = ContinuousConfig()
        conf.observation_mode = observation_mode
        conf.spacing_size = spacing_size
        conf.tile_size = tile_size
        conf.ball_size = ball_size
        conf.t_max = t_max
        conf.ball_start = ball_start
        conf.seq = seq
        conf.task_cues = task_cues
        conf.seed = seed

        return Numpad2DContinuous(conf)

    def test_step_before_reset_failure(self):
        env = self.init_3x3()
        with pytest.raises(RuntimeError):
            env.step(np.array([1, 1], dtype=float))

    def test_task_cues(self):
        # Partial task cue with everything hidden
        env = self.init_3x3(task_cues=True, seed=1337)
        obs = env.reset()
        assert obs[4:].sum() == 0
        assert obs[4:].sum() != env.seq_len

        # Partial task cue with only some hidden
        env = self.init_3x3(task_cues=True, seed=1338)
        obs = env.reset()
        assert obs[4:].sum() == 1

        # Partial task cue with everything shown
        env = self.init_3x3(task_cues=True, seed=1470)
        obs = env.reset()
        assert obs[4:].sum() == 3

        # Task cues disabled
        env = self.init_3x3(task_cues=False, seed=1470)
        obs = env.reset()
        assert obs[4:].sum() == 0

        # greyscale task cues
        env = self.init_3x3(observation_mode="greyscale_array", task_cues=True, seed=1470)
        obs = env.reset()
        obs_unique_values = (np.unique(obs) * 255.000000001).astype(np.uint8) # Fix minor float issue of 204 actually becoming 203 on floor/uint8 cast
        greyscale_light_on = np.mean(env.LIGHT_ON).astype(np.uint8)
        assert (obs_unique_values == greyscale_light_on).any()

        # rgb task cues
        env = self.init_3x3(observation_mode="rgb_array", task_cues=True, seed=1470)
        obs = env.reset()
        for i, channel_light_color in enumerate(env.LIGHT_ON):
            channel_unique_values = (np.unique(obs[i]) * 255.000000001).astype(np.uint8)
            assert (channel_unique_values == channel_light_color).any()

    def test_ball_start_or_seq_none(self):
        with pytest.raises(ValueError):
            self.init_3x3(ball_start=None)

        with pytest.raises(ValueError):
            self.init_3x3(seq=None)

    def test_invalid_action(self):
        env = self.init_3x3(observation_mode="ram")
        env.reset()

        with pytest.raises(ValueError):
            env.step(np.array([100, 100], dtype=float))

        with pytest.raises(ValueError):
            env.step(np.array([-2, -2], dtype=float))

    def test_observation_modes_initialization(self):
        self.init_3x3(observation_mode="ram")
        self.init_3x3(observation_mode="rgb_array")
        self.init_3x3(observation_mode="greyscale_array")

        with pytest.raises(ValueError):
            self.init_3x3(observation_mode="some invalid observation mode")

    def test_ram_observation_mode(self):
        env = self.init_3x3(observation_mode="ram",
                            ball_start=[125, 375],
                            seq=[0, 1, 2])
        obs = env.reset()
        assert len(obs) == 13  # 4 (ball pos. vec.) + 9 (num. tiles)
        assert (obs[:2] >= 0).all() and (
            obs[:2] <= 1).all()  # ball position is normalized
        assert (obs[2:] == 0).all()  # all lights are off

        env.ball_pos = np.array([125, 125], dtype=float)
        obs, rew, _, _ = env.step(np.array([0, 0], dtype=float))
        assert rew == 1
        assert len(obs) == 13
        assert (obs[:2] >= 0).all() and (obs[:2] <= 1).all()
        assert (obs[5:] == 0).all()
        assert obs[4] == 1  # First of sequence lights up

    def test_rgb_observation_mode(self):
        env = self.init_3x3(observation_mode="rgb_array")
        obs = env.reset()

        assert obs.shape[0] == 3  # color channel x width x height
        assert (obs >= 0).all() and (obs <= 1).all()

    def test_greyscale_observation_mode(self):
        env = self.init_3x3(observation_mode="greyscale_array")
        obs = env.reset()

        assert obs.shape[0] == 1
        assert (obs >= 0).all() and (obs <= 1).all()

    def test_timeup_done(self):
        env = self.init_3x3()
        env.reset()
        done = False

        for _ in range(50):
            _, _, done, _ = env.step(np.array([-1., -1.]))

        assert done == True

    def test_rewards_whole_sequence(self):
        env = self.init_3x3(seq=[0, 1, 2])
        env.reset()

        # check that we get reward for first correct press
        env.ball_pos = np.array((125., 125.))
        obs, rew, _, _ = env.step(np.array((0., 0.)))
        # check that the first light in the observation is on
        assert obs[4] == 1
        assert rew == 1

        # check that there is no reward in second step on first tile
        # but that it still lights up
        env.ball_pos = np.array((125., 130.))
        obs, rew, _, _ = env.step(np.array((0., 0.)))
        assert obs[4] == 1
        assert rew == 0

        # check that there is reward on second tile
        env.ball_pos = np.array((125., 375.))
        _, rew, _, _ = env.step(np.array((0., 0.)))
        assert rew == 1

        # check that there is no reward while standing on ground
        env.ball_pos = np.array((300., 300.))
        _, rew, _, _ = env.step(np.array((0., 0.)))
        assert rew == 0

        # check that there is reward on third tile
        env.ball_pos = np.array((125., 601.))
        _, rew, _, _ = env.step(np.array((0., 0.)))
        assert rew == 1

    def test_rewards_redo_sequence_last_to_first(self):
        """
        Completing the sequence, then moving directly from last tile to first tile of sequence
        """
        env = self.init_3x3(seq=[0, 1, 2])
        env.reset()

        # perform sequence once (as tested in test_rewards_whole_sequence)
        env.ball_pos = np.array((125., 125.))
        env.step(np.array((0., 0.)))
        env.ball_pos = np.array((125., 375.))
        env.step(np.array((0., 0.)))
        env.ball_pos = np.array((125., 601.))
        env.step(np.array((0., 0.)))

        # then start over and check rewards
        env.ball_pos = np.array((125., 125.))
        _, rew, _, _ = env.step(np.array((0., 0.)))
        assert rew == 1
        env.ball_pos = np.array((125., 375.))
        _, rew, _, _ = env.step(np.array((0., 0.)))
        assert rew == 1
        env.ball_pos = np.array((125., 601.))
        _, rew, _, _ = env.step(np.array((0., 0.)))
        assert rew == 1

    def test_rewards_redo_sequence_intermediate(self):
        """
        Completing the sequence, then moving to a tile that is not the first tile of the sequence
        and then to the first til of the sequence
        """
        env = self.init_3x3(seq=[0, 1, 2])
        env.reset()

        # perform sequence once (as tested in test_rewards_whole_sequence)
        env.ball_pos = np.array((125., 125.))
        env.step(np.array((0., 0.)))
        env.ball_pos = np.array((125., 375.))
        env.step(np.array((0., 0.)))
        env.ball_pos = np.array((125., 601.))
        env.step(np.array((0., 0.)))

        # step on tile which is not the first in the sequence
        env.ball_pos = np.array((375., 375.))
        env.step(np.array((0., 0.)))

        # then start over and check rewards
        env.ball_pos = np.array((125., 125.))
        _, rew, _, _ = env.step(np.array((0., 0.)))
        assert rew == 1
        env.ball_pos = np.array((125., 375.))
        _, rew, _, _ = env.step(np.array((0., 0.)))
        assert rew == 1
        env.ball_pos = np.array((125., 601.))
        _, rew, _, _ = env.step(np.array((0., 0.)))
        assert rew == 1

    def test_reset_lights(self):
        env = self.init_3x3(seq=[0, 1, 2])
        env.reset()

        env.ball_pos = np.array([125., 125.])  # tile 0
        _, rew, _, _ = env.step(np.array([0., 0.]))
        assert rew == 1
        assert env.next_seq_ind == 1
        assert env.max_seq_ind == 0

        env.ball_pos = np.array([375., 125.])  # tile 3
        _, rew, _, _ = env.step(np.array([0., 0.]))
        assert rew == 0
        assert env.next_seq_ind == 0  # Lights reset
        assert env.max_seq_ind == 0

    def test_reset(self):
        np.random.seed(123)

        # If the ball pos. and sequence are specified then the same are used in
        # all episodes, across resets.
        env = self.init_3x3()
        env.reset()
        ball_start1, seq1 = env.ball_pos, env.seq

        env.reset()
        ball_start2, seq2 = env.ball_pos, env.seq

        assert (ball_start1 == ball_start2).all()
        assert seq1 == seq2

        # If ball pos. and sequence are not specified (i.e. both are None), then
        # they should (most likely) be different across resets of the environment
        env = self.init_3x3(ball_start=None, seq=None)
        env.reset()
        ball_start3, seq3 = env.ball_pos, env.seq

        env.reset()
        ball_start4, seq4 = env.ball_pos, env.seq

        assert (ball_start3 != ball_start4).all()
        assert seq3 != seq4

    def test_velocity_change(self):
        env = self.init_3x3(seq=[0, 1, 2])
        env.reset()

        env.step(np.array((1., 1.)))
        assert (env.ball_pos == np.array((5., 5.))).all()

        env.step(np.array((1., -1.)))
        assert (env.ball_pos == np.array((15., 5.))).all()

        env.step(np.array((0., 0.)))
        assert (env.ball_pos == np.array((25., 5.))).all()

        env.step(np.array((-1., -1.)))
        assert (env.ball_pos == np.array((30., 0.))).all()

        # ball hits the wall in this step, so y-coordinate stays at 0
        env.step(np.array((-1., 0.)))
        assert (env.ball_pos == np.array((30., 0.))).all()
        assert (env.ball_velocity == np.array((0., 0.))).all()
