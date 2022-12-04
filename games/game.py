
import numpy as np

STATE_WAITING = 0
STATE_IN_PROGRESS = 1
STATE_GAME_OVER = 2

class Game:
    """
    A simple game to be interacted with.
    A game has:
     - A `score`
     - A `game_state` which is one of `{waiting, in_progress, done}`
     - A `render_state` which is a grid of numerical values to draw

    A game can either generate observations with `observe()`
    or act through `act(action)`. The render state can be recomputed
    and returned with `draw()`
    """

    def __init__(self, height = 32, width = 64):
        """
        Initialize the game and its attributes
        """
        self.height = height
        self.width = width
        self.render_state: np.NDArray = np.zeros((height, width), dtype=np.int8)
        self.reset()

    def reset(self):
        """
        Reset the game to its initial state
        """
        self.score = 0
        self.reward = 0
        self.game_state = STATE_IN_PROGRESS
        self.render_state.fill(0)

    def _get_pixel(self, px):
        """
        Return the char and color index used to represent the pixel at
        coordinates `px`
        """
        return '█', 0

    def draw(self):
        """
        Update the game's `render_state`and return it
        """
        pass

    def observe(self):
        """
        Generate and returns observations about the game's current state
        """
        pass

    def act(self, action):
        """
        Perform an action as a player
        """
        pass