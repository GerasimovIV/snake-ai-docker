import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer, Linear_QNet_Scanner
from helper import plot
import argparse
import torch.nn.functional as F
import torch.nn as nn
from agents import *

from pathlib import Path

FILE = Path(__file__).resolve()


            


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    
#     agent = Agent()
#     agent = AgentScaner()
    agent = AgentSNAKE()
    

    game = SnakeGameAI(food_n=1)
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

            
def play(agent, game):

    statistic = []
    mean_scores = []
    record = 0
    total_score = 0
    
    while True:
        # get old state
        state = agent.get_state(game)

        # get move
        move = agent.get_action_test(state)

        # perform move and get new state
        reward, done, score = game.play_step(move)

        if done:
            
            statistic.append(score)
            
            game.reset()
            agent.n_games += 1

            if score > record:
                record = score

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            total_score += score
            mean_score = total_score / agent.n_games
            mean_scores.append(mean_score)
            plot(statistic, mean_scores, 'Testing...')
    
if __name__ == '__main__':
    
    train()