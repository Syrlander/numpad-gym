from rl_thesis.environments import numpad_base


class Config(numpad_base.Config):

    # whether or not to flatteen state before returning, this changes the observation space
    flatten_state = False

    def __init__(self):
        super().__init__()
        [
            setattr(self, k, v) for k, v in vars(Config).items()
            if not k.startswith("_")
        ]