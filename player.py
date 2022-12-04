from net.PlayerNet import PlayerNet
from games.game import Game

import numpy as np
import random
import torch
from collections import deque


class Player():
    """
    A player
    """
    def __init__(self, game: Game, input_n, action_n, hidden_n = 128, batch_size = 12, learn_every=3, lr=0.0_25):
        self.game = game
        self.n_games = 0
        self.curr_step = 0

        self.input_n = input_n
        self.action_n = action_n
        self.hidden_n = hidden_n

        self.batch_size = batch_size
        
        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.learn_every = learn_every

        self.gamma = 0.9
        self.memory = deque(maxlen=100_000)
        self.model = PlayerNet(input_n, hidden_n, action_n)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = torch.nn.SmoothL1Loss()

    def choose_action(self, state):
        """
        Choose action based on the state.
        The action might be chosen at random (exploration),
        or computed with the state through the model (exploitation)
        """
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_n)
        # EXPLOIT
        else:
            action_values = self.model(torch.tensor(state, dtype=torch.float))
            action_idx = torch.argmax(action_values, axis=1).item()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_idx

    def memorize(self, state, next_state, action, reward, done, score):
        """
        Store an experience in memory
        """
        self.memory.append((state, next_state, action, reward, done, score))

    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        if len(self.memory) < self.batch_size:
            samples = self.memory
        else:
            samples = random.sample(self.memory, self.batch_size)

        state, next_state, action, reward, done, score = zip(*samples)
        return state, next_state, action, reward, done, score

    def learn(self):

        if (self.curr_step % self.learn_every != 0):
            return None

        state, next_state, action, reward, done, score = self.recall()

        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: Predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # 2: Q_new = r + y * max(next predicted Q value) -> only do this if not done
        self.optimizer.zero_grad()
        loss = self.loss_fn(target, pred)
        loss.backward()

        self.optimizer.step()

        return loss
