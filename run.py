import torch
from games.snake import Snake
from games.game import STATE_GAME_OVER
from player import Player
from utils.display import Display
from utils.logger import MetricLogger

from pathlib import Path
import datetime

from random import randint

if (__name__ == "__main__"):
    game = Snake()
    disp = Display(game)

    use_mps = torch.backends.mps.is_available()

    save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True)

    player = Player(game, len(game.observe()), 3)

    logger = MetricLogger(save_dir)

    episodes = 100

    for e in range(episodes):

        game.reset()

        state = game.observe()

        while True:

            # Run agent on the state
            action = player.choose_action(state)

            # Agent performs action
            reward, score, done = game.act(action)
            next_state = game.observe()

            # Remember
            player.memorize(state, next_state, action, reward, done, score)

            # Learn
            loss = player.learn()

            # Logging
            #logger.log_step(reward, loss, 0)
            disp.draw()

            # Update state
            state = next_state

            if done:
                break

        logger.log_episode()

        if (e % 20 == 0):
            logger.record(episode=e, epsilon=player.exploration_rate, step=player.curr_step)


    player.model.save()
