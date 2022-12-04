import numpy as np
import curses
from random import randint

from games.game import Game, STATE_GAME_OVER, STATE_IN_PROGRESS

w = 32
h = 16

class Snake(Game):
    def __init__(self, height = h, width = w):
        super().__init__(height, width)
        self.reset()

    def reset(self):
        super().reset()
        self.spawn_snake()
        self.spawn_food()

    def observe(self):
        state = [
            # Danger to the left
            self.check_collisions(self.generate_pixel(0)),
            # Danger to the right
            self.check_collisions(self.generate_pixel(1)),
            # Danger in front
            self.check_collisions(self.generate_pixel(2)),

            
            # Direction left
            self.angle == 0,
            # Direction right
            self.angle == 180,
            # Direction up
            self.angle == 90,
            # Direction down
            self.angle == 270,


            # Location of food
            # Food left
            self.food[1] < self.snake[0][1],
            # Food right
            self.food[1] > self.snake[0][1],
            # Food up
            self.food[0] > self.snake[0][0],
            # Food down
            self.food[0] < self.snake[0][0]
        ]

        return np.array(state, dtype=np.int8)

    def act(self, action):
        self.reward = 0
        if (self.game_state == STATE_IN_PROGRESS):
            self.snake.insert(0, self.generate_pixel(action))
            if self.snake[0] == self.food:
                self.score += 1
                self.reward = 10
                self.spawn_food()
            else:
                self.snake_remove()
            if (self.check_collisions()):
                self.reward = -10
                self.game_state = STATE_GAME_OVER

        return self.reward, self.score, self.game_state == STATE_GAME_OVER


    def spawn_snake(self, init_length = 3):
        margin = 5
        x = randint(margin, self.render_state.shape[1] - margin)
        y = randint(margin, self.render_state.shape[0] - margin)
        self.snake = []
        vertical = randint(0, 1) == 0
        for i in range(min(init_length, margin - 1)):
            px = [y + i, x] if vertical else [y, x + i]
            self.snake.insert(0, px)

        self.angle = 90 if vertical else 0

    def generate_pixel(self, action):
        """
        Generate the coordinates of a
        new snake segment to add,
        based on current head coordinates, direction,
        and action
        """
        p = [self.snake[0][0], self.snake[0][1]]
        angle = 0
        # w: forward
        if action == 2:
            angle = 0
        # a: left
        elif action == 0:
            angle = -90
        # d: right
        elif action == 1:
            angle = 90

        self.angle = (self.angle + angle) % 360

        return [p[0] + int(np.sin(np.deg2rad(self.angle))),
                     p[1] + int(np.cos(np.deg2rad(self.angle)))]

    def snake_remove(self):
        self.snake.pop()

    def check_collisions(self, new_point=[0,0]):
        y = self.snake[0][0] + new_point[0]
        x = self.snake[0][1] + new_point[1]
        return (y == 0 or
                y == self.render_state.shape[0] or
                x == 0 or
                x == self.render_state.shape[1] or
                [y, x] in self.snake[1:])

    def spawn_food(self):
        food = []
        while food == []:
            food = [randint(1, self.render_state.shape[0] - 1), randint(1, self.render_state.shape[1] - 1)]
            if food in self.snake: food = []
        self.food = food

    def _get_pixel(self, px):
        p = self.render_state[px[0], px[1]]
        if (p == 1):
            return '█', curses.COLOR_WHITE
        elif (p == -1):
            return 'o', curses.COLOR_YELLOW
        elif (p == 2):
            return '█', curses.COLOR_GREEN

    def draw(self):
        self.render_state.fill(0)
        self.render_state[self.snake[0][0], self.snake[0][1]] = 2
        for y, x in self.snake[1:]:
            self.render_state[y, x] = 1
        self.render_state[self.food[0], self.food[1]] = -1
        return self.render_state