import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer, Linear_QNet_Scanner, WrappedLSTM, QTrainerLSTM
from helper import plot
import argparse
import torch.nn.functional as F
import torch.nn as nn


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
#         self.model = Linear_QNet_Scanner(11, 256, 3, activations=[F.relu, nn.Identity()]) # Linear_QNet(11, 256, 3)
        
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN
        
        game.foods.sort(key = lambda x: np.sqrt( (x.x - head.x) ** 2) + (x.y - head.y) ** 2)
        food = game.foods[0]
        

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            food.x < game.head.x,  # food left
            food.x > game.head.x,  # food right
            food.y < game.head.y,  # food up
            food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
        
    def get_action_test(self, state):
        final_move = [0,0,0]
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0)
        move = torch.argmax(prediction).item()
        final_move[move] = 1
        
        return final_move
        

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move
    
    def load_model(self, weights_path):
        
        self.model.load_state_dict(torch.load(weights_path))
        
        
    
class AgentScaner(Agent):
    def __init__(self):
        
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        
        self.model = Linear_QNet_Scanner(11, 256, 3, activations=[nn.ReLU(), nn.Identity()])
        
        print(self.model)
        
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

        
    def _get_left_from_cur_direction(self, direction):
        if direction == Direction.LEFT:
            return Direction.DOWN 
        
        elif direction == Direction.RIGHT:
            return Direction.UP
        
        elif direction == Direction.UP:
            return Direction.LEFT
        
        return Direction.RIGHT
    
    
    def _get_right_from_cur_direction(self, direction):
        if direction == Direction.LEFT:
            return Direction.UP
        
        elif direction == Direction.RIGHT:
            return Direction.DOWN
        
        elif direction == Direction.UP:
            return Direction.RIGHT
        
        return Direction.LEFT
    
    
    def _get_straight_from_cur_direction(self, direction):
        return direction
        
    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN
        
        cur_dir = Direction.LEFT
        
        if dir_r:
            cur_dir = Direction.RIGHT
        if dir_u:
            cur_dir = Direction.UP
        if dir_d:
            cur_dir = Direction.DOWN
            
        game.foods.sort(key = lambda x: np.sqrt( (x.x - head.x) ** 2) + (x.y - head.y) ** 2)

        
        food = game.foods[0]
        
        state = [
            # Danger 
#             self._get_free_ray_len(self._get_left_from_cur_direction(cur_dir), game),
#             self._get_free_ray_len(self._get_right_from_cur_direction(cur_dir), game),
#             self._get_free_ray_len(self._get_straight_from_cur_direction(cur_dir), game),
            
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            food.x < game.head.x,  # food left
            food.x > game.head.x,  # food right
            food.y < game.head.y,  # food up
            food.y > game.head.y  # food down
#             game.food.x - game.snake[0].x,
#             game.food.y - game.snake[0].y,
            ]

        return np.array(state, dtype=float)
    
    def _get_free_ray_len(self, direction, game) -> int:
        
        head = game.snake[0]
        
        vec_l = Point(- 20, 0)
        vec_r = Point(+ 20, 0)
        vec_u = Point(0, - 20)
        vec_d = Point(0, + 20)
        
        dir_l = direction == Direction.LEFT
        dir_r = direction == Direction.RIGHT
        dir_u = direction == Direction.UP
        dir_d = direction == Direction.DOWN
        
        
        if dir_l:
            add_vec = vec_l
        
        elif dir_r:
            add_vec = vec_r
            
        elif dir_u:
            add_vec = vec_u
        
        elif dir_d:
            add_vec = vec_d
        
        i = 1.
        
        next_pos = head
        
        denominator = 1.
        
        if dir_l or dir_r:
            denominator = game.w / 20
            
        elif dir_u or dir_d:
            denominator = game.h / 20
        
        while True:
            
            next_pos += i * add_vec
            
            if game.is_collision(next_pos):
                
                return i / denominator
        
            i += 1

            
class AgentSNAKE(Agent):

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        
        self.model = WrappedLSTM(input_size=11, 
                                 hidden_size=256, 
                                 proj_size=3, 
                                 bidirectional=False, 
                                 batch_first=True)
        
        self.trainer = QTrainerLSTM(self.model, lr=LR, gamma=self.gamma)
        
    def direction_btw_points(self, p1, p2):
        dv = p2 - p1
        
        if dv.x > 0:
            return Direction.RIGHT
        elif dv.x < 0:
            return Direction.LEFT
        
        if dv.y > 0:
            return Direction.UP
        elif dv.y < 0:
            return Direction.DOWN
            
        return None
    
    def get_state_i(self, game, i):
        
        head = game.snake[i]
        
        if i == 0:
            next_direction = game.direction
        else:
            next_direction = self.direction_btw_points(head, game.snake[i - 1])
        
        
            
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = next_direction == Direction.LEFT
        dir_r = next_direction == Direction.RIGHT
        dir_u = next_direction == Direction.UP
        dir_d = next_direction == Direction.DOWN
        
        game.foods.sort(key = lambda x: np.sqrt( (x.x - head.x) ** 2) + (x.y - head.y) ** 2)
        food = game.foods[0]
        

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            food.x < game.head.x,  # food left
            food.x > game.head.x,  # food right
            food.y < game.head.y,  # food up
            food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)


    def get_state(self, game):
        
        states = []
        for i in range(len(game.snake) - 1, -1, -1):
            states.append(self.get_state_i(game, i))

        return np.vstack(states)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
        
    def get_action_test(self, state):
        final_move = [0,0,0]
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0)
        move = torch.argmax(prediction).item()
        final_move[move] = 1
        
        return final_move
        

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move
    
    def load_model(self, weights_path):
        
        self.model.load_state_dict(torch.load(weights_path))