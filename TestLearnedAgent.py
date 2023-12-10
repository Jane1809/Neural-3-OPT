import argparse
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from DataGenerator import TSPDataset
from tqdm import tqdm
from TSPEnvironment import TSPInstanceEnv, VecEnv
from ActorCriticNetwork import ActorCriticNetwork

import pickle
import time

parser = argparse.ArgumentParser(description='TSPNet')

# ----------------------------------- Data ---------------------------------- #
parser.add_argument('--test_size',
                    default=10000, type=int, help='Test data size')
parser.add_argument('--test_from_data',
                    default=True,
                    action='store_true', help='Render')
parser.add_argument('--n_points',
                    type=int, default=100, help='Number of points in TSP')
# ---------------------------------- Train ---------------------------------- #
parser.add_argument('--n_steps',
                    default=2000,
                    type=int, help='Number of steps in each episode')
parser.add_argument('--render',
                    default=True,
                    action='store_true', help='Render')
# ----------------------------------- GPU ----------------------------------- #
parser.add_argument('--gpu',
                    default=True, action='store_true', help='Enable gpu')
# --------------------------------- Network --------------------------------- #
parser.add_argument('--input_dim',
                    type=int, default=2, help='Input size')
parser.add_argument('--embedding_dim',
                    type=int, default=128, help='Embedding size')
parser.add_argument('--hidden_dim',
                    type=int, default=128, help='Number of hidden units')
parser.add_argument('--n_rnn_layers',
                    type=int, default=1, help='Number of LSTM layers')
parser.add_argument('--n_actions',
                    type=int, default=3, help='Number of nodes to output')
parser.add_argument('--graph_ref',
                    default=False,
                    action='store_true',
                    help='Use message passing as reference')

# --------------------------------- Misc --------------------------------- #
parser.add_argument('--load_path', type=str,
    default='/home/suijingyan/sjy_data/suijingyan/3-opt/Neural-3-OPT/models/Tsp20_3-opt-3-type-KGcn-FilmV3-cattosum-3film-speedup-sum-k8/pg-695667f45eb143d1ac90ef2ce6becb7a-TSP20-epoch-198.pt')
parser.add_argument('--data_dir', type=str, default='data')

args = parser.parse_args()

if args.gpu and torch.cuda.is_available():
    USE_CUDA = True
    print('Using GPU, %i devices available.' % torch.cuda.device_count())
else:
    USE_CUDA = False

# loading the model from file
if args.load_path != '':
    print('  [*] Loading model from {}'.format(args.load_path))

    model = ActorCriticNetwork(args.input_dim,
                               args.embedding_dim,
                               args.hidden_dim,
                               args.n_points,
                               args.n_rnn_layers,
                               args.n_actions,
                               args.graph_ref)
    checkpoint = torch.load(os.path.join(os.getcwd(), args.load_path))
    policy = checkpoint['policy']
    model.load_state_dict(policy)

# Move model to the GPU
if USE_CUDA:
    torch.cuda.set_device(torch.device('cuda:3'))
    model.cuda()

# if args.test_from_data:
    # test_data = TSPDataset(dataset_fname=os.path.join(args.data_dir,
    #                                                   'TSP{}-data-test.json'
    #                                                   .format(args.n_points)),
    #                        num_samples=args.test_size, seed=1234)
if args.test_from_data:
    test_data = TSPDataset(dataset_fname=os.path.join(args.data_dir,
                                                      'att-TSP{}-data-test.json'
                                                      .format(args.n_points)),
                           num_samples=args.test_size, seed=1234)


test_loader = DataLoader(test_data,
                         batch_size=args.test_size,
                         shuffle=False,
                         num_workers=6)

test_history = {
    "data": test_data.ids,
    # "tours": [],
    "dis": [],
    "best_dis": []
}

# run agent
model = model.eval()
rewards = []
best_distances = []
step_best_distances = []
distances = []
initial_distances = []
distances_per_step = []

t_s=time.time()
for batch_idx, batch_sample in enumerate(test_loader):
    
    b_sample = batch_sample.clone().detach().numpy()
    sum_reward = 0
    env = VecEnv(TSPInstanceEnv,
                 b_sample.shape[0],
                 args.n_points)
    state, initial_distance, best_state = env.reset(b_sample)
    t = 0
    hidden = None
    pbar = tqdm(total=args.n_steps)
    # render_idx = 0

    # batch_tours = []

    while t < args.n_steps:
        if args.render:
            env.render()
        state = torch.from_numpy(state).float()
        # if t == 0:
        #     print(state[0])
        best_state = torch.from_numpy(best_state).float().cuda()
        if USE_CUDA:
            state = state.cuda()
        with torch.no_grad():
            _, action, _, _, _, hidden, _, opt_action, _, _ = model(state, best_state, hidden, env=env)
        action = action.cpu().numpy()
        opt_action = opt_action.cpu().numpy()

        # print(opt_action)
        # print("opts: " + " ".join([str(_i)for _i in action[render_idx]]), flush=True)
        state, reward, _, best_distance, distance, best_state = env.step(action, opt_action)


        # for _env in env.envs:
        #     batch_tours.append(_env.keep_tour)

        sum_reward += reward
        t += 1
        step_best_distances.append(np.mean(best_distance)/10000)
        distances_per_step.append(best_distance)
        pbar.update(1)
    pbar.close()

    # test_history["tours"].append(batch_tours)


    dis, best_dis = [], []
    for _env in env.envs:
        dis.append(_env.hist_current_distance)
        best_dis.append(_env.hist_best_distance)

    test_history["dis"].append(dis)
    test_history["best_dis"].append(best_dis)


    rewards.append(sum_reward)
    best_distances.append(best_distance)
    distances.append(distance)
    initial_distances.append(initial_distance)
avg_reward = np.mean(rewards)
avg_best_distances = np.mean(best_distances)
avg_initial_distances = np.mean(initial_distances)
gap = ((avg_best_distances/10000/np.mean(test_data.opt))-1)*100
t_e = time.time()


print('Initial Cost: {:.5f} Best Cost: {:.5f} Opt Cost: {:.5f} Gap: {:.2f} %'.format(
    avg_initial_distances/10000, avg_best_distances/10000, np.mean(test_data.opt), gap))
print((t_e - t_s)/60)


# with open("./improve_his_3opt_3_T200s2000.pkl", 'wb') as f:
#     pickle.dump(test_history, f)
