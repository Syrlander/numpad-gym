from rl_thesis.environments.numpad_continuous import Config as ContinuousConfig

class Config(ContinuousConfig):

  def __init__(self):
      super().__init__()
      [
          setattr(self, k, v) 
          for k, v in vars(Config).items() 
              if not k.startswith("_")
      ]