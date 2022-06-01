from rl_thesis.environments import numpad_base


class Config(numpad_base.Config):
    """
    spacing_size: 
        number of pixels between each tile (and between tiles and boundary)
    tile_size: 
        height and width of each tile
    ball_size:
        Height and width of ball, should be odd to ensure ball is centered around actual position
    """
    # Available observation modes (spaces):
    # * "rgb_array"
    # * "greyscale_array"
    # * "ram"
    observation_mode = "ram"

    spacing_size = 5
    tile_size = 10
    ball_size = 5
    t_max = 200

    def __init__(self):
        super().__init__()
        [
            setattr(self, k, v) for k, v in vars(Config).items()
            if not k.startswith("_")
        ]