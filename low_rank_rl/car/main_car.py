import numpy as np
import torch
import gym

import argparse
import random
import json
from distutils.util import strtobool
from pprint import pprint

from utils import Mapper, MhRL


def parse_args():
    parser = argparse.ArgumentParser(description='RL task: Mountain Car')

    parser.add_argument('--seed', type=int, default=1)

    # algorithm parameters
    parser.add_argument('--alg', type=str, default='')
    parser.add_argument('--episodes', type=int, default=40000, help='K')
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--alpha', type=float, default=0.0003, help='learning rate')

    

    # test parameters
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument('--test_fre', type=int, default=10)

    # parameters specific to low-rank Mh learning
    parser.add_argument('--rank', type=int, default=10)


    args = parser.parse_args()
    return args


def generate_para(parameters_file):
    with open(parameters_file) as j:
        parameters = json.loads(j.read())

    mapping = Mapper()
    state_map, state_reverse_map = mapping.get_state_map(parameters["step_state"], parameters["decimal_state"])
    action_map, action_reverse_map = mapping.get_action_map(parameters["step_action"], parameters["decimal_action"])

    return parameters, state_map, state_reverse_map, action_map, action_reverse_map



def wandb_train(config=None):
    """ Initialize algorithms based on command line argument """
    args = parse_args()

    with wandb.init(config=config, save_code=True):
        config = wandb.config
        args.seed = config.seed
        args.alg = config.alg
        args.rank = config.rank

        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)


        """Environment Initialization"""
        env = gym.make("MountainCarContinuous-v0")

        """Algorithm Initialization"""
        if args.alg == 'Mhlearning':
            parameters_file = "low_rank_rl/car/experiments/exp_1_Mh_learning.json"
            parameters, state_map, state_reverse_map, action_map, action_reverse_map = generate_para(parameters_file)
            n_states = len(state_map)
            n_actions = len(action_map)

            args.alpha = 0.0003

            learner = MhRL(env=env,
                        state_map=state_map,
                        action_map=action_map,
                        state_reverse_map=state_reverse_map,
                        action_reverse_map=action_reverse_map,
                        n_states=n_states,
                        n_actions=n_actions,
                        decimal_state=parameters["decimal_state"],
                        decimal_action=parameters["decimal_action"],
                        step_state=parameters["step_state"],
                        step_action=parameters["step_action"],
                        episodes=args.episodes,
                        max_steps=parameters["max_steps"],
                        alpha=args.alpha,
                        gamma=parameters["gamma"],
                        r=args.rank,
                        args=args)
            
        learner.train()
        


if __name__ == '__main__':
    import wandb

    sweep_config = {
        'method': 'grid'
        }

    parameters_dict = {
        'seed':{
            'values':[1,2,3,4,5]#,2,3,4,5,6,7,8,9,10
            },
        'alg': {
            'values': ['Mhlearning']
            },
        'rank':{
            'values': [5]#10,15
        }
    }
    
    sweep_config['parameters'] = parameters_dict
    pprint(sweep_config)

    sweep_id = wandb.sweep(sweep_config, project="Test Low-rank Car")
    wandb.agent(sweep_id, wandb_train)

