import argparse
import os
import sys
import pickle
import numpy as np
import torch
from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize

parser = argparse.ArgumentParser(description='Save expert trajectory')
parser.add_argument('--env-name', default="Hopper-v2", help='name of the environment to run')
parser.add_argument('--load-dir', default='./trained_models/', help='name of the expert model')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--max-step-num', type=int, default=1000,
                    help='maximal number of main iterations (default: 1000)')
parser.add_argument('--traj-num', type=int, default=50000,
                    help='number of trajectoires (default: 50000)')
parser.add_argument( '--non-det', action='store_true', default=False,
                    help='whether to use a non-deterministic policy')
args = parser.parse_args()
args.det = not args.non_det

env = make_vec_envs(args.env_name, args.seed, 1, None, None, device='cpu', allow_early_resets=False)

actor_critic, ob_rms = torch.load(args.load_dir, map_location='cpu')

vec_norm = get_vec_normalize(env)
if vec_norm is not None:
    vec_norm.eval()
    vec_norm.ob_rms = ob_rms

recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
masks = torch.zeros(1, 1)
states = []
actions = []
rewards = []
lens = [] 

for i in range(args.traj_num):
    s_list = []
    a_list = []
    r_list = []

    reward_episode = 0
    obs = env.reset() 
    for t in range(args.max_step_num):
        with torch.no_grad():
            value, action, _, recurrent_hidden_states = actor_critic.act(
                obs, recurrent_hidden_states, masks, deterministic=args.det)
        s_list.append(obs[0].float())
        a_list.append(action[0].float())
        # Obser reward and next obs
        obs, reward, done, _ = env.step(action)
        r_list.append(reward[0][0].float())

        masks.fill_(0.0 if done else 1.0)
        reward_episode += reward[0][0].item()
        if done:
            break
    if t+1 == args.max_step_num: 
        states.append(torch.stack(s_list))
        actions.append(torch.stack(a_list))
        rewards.append(torch.stack(r_list))
        lens.append(t+1)

    print('Episode {}\t reward: {:.2f}'.format(i, reward_episode))
print(lens)
expert_traj = {
    'states': torch.stack(states),
    'actions': torch.stack(actions),
    'rewards': torch.stack(rewards),
    'lengths': torch.LongTensor(lens)
}
torch.save(expert_traj, './expert_traj/{}_expert_traj.p'.format(args.env_name))
