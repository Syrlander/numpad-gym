class Config:
  """
  numpad_size: 
      number of tiles along rows and columns
  seq_len:
      Length of the sequence to find
  t_max: 
      the maximum number of timesteps to run episode for
  """
  numpad_size = 3
  seq_len = 4
  t_max = 500
  ball_start = None
  seq = None
  task_cues : bool = False
  task_cue_threshold : float = 0.8
  seed : int = None

  def __init__(self):
        [
            setattr(self, k, v) 
            for k, v in vars(Config).items() 
                if not k.startswith("_")
        ]