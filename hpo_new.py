import argparse
import subprocess
import re
import numpy as np
import time
from sklearn.model_selection import ParameterGrid, RandomizedSearchCV, GridSearchCV
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt

import random
import numpy as np
import torch

def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set seed globally
set_all_seeds(42)


global losses
losses =[]
def run_main(params):
    #cmd = [
    #     'python', 'main.py',
    #     '--lr', str(params['lr']),
    #     '--batch-size', str(params['batch_size']),
    #     '--n-conv', str(params['num_layers']),
    #     '--h-fea-len', str(params['hidden_dim']),
    #     '--dropout-fraction', str(params['dropout_fraction']),
    #     '--weight-decay', str(params['weight_decay']),
    #     'data/sample-regression'
    # ]
    cmd = [
        'python', 'main.py',
        '--batch-size', str(params['batch_size']),
        '--lr', str(params['lr']),
        '--weight-decay', str(params['weight_decay']),
        '--atom-fea-len', str(params['atom_fea_len']),
        '--h-fea-len', str(params['h_fea_len']),
        '--n-conv', str(params['n_conv']),
        '--n-h', str(params['n_h']),
        'data/sample-regression'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # print("STDOUT:\n", result.stdout)
    # print("STDERR:\n", result.stderr)
    match = re.search(r"Validation Loss ([0-9.]+)", result.stdout)
    print(match)
    if match:
        losses.append(float(match.group(1)))
        return float(match.group(1))
    else:
        print("Failed to retrieve validation loss. Check main.py output.")
        return float('inf')

def random_search(trials=10):
    param_space = {
        'lr': np.logspace(-8, -3, 10),
        'batch_size': [16, 32, 64],
        'weight_decay': np.logspace(-6, -2, 10),
        'atom_fea_len': [64, 128, 256, 512],
        'h_fea_len': [64, 128, 256, 512],
        'n_conv': range(1, 6),
        'n_h': range(1, 6)
    }
    best_loss = float('inf')
    best_params = None
    for i in range(trials):
        params = {k: np.random.choice(v) if isinstance(v, list) else np.random.uniform(min(v), max(v)) for k, v in param_space.items()}
        loss = run_main(params)
        print(f"Trial {i+1}, Loss: {loss}, Params: {params}")
        if loss < best_loss:
            best_loss = loss
            best_params = params
    print(f"Best Loss: {best_loss}, Best Params: {best_params}")

def grid_search():
    param_space = {
        'lr': np.logspace(-8, -3, 5),
        'batch_size': [16, 32, 64],
        'weight_decay': np.logspace(-6, -2, 5),
        'atom_fea_len': [64, 128, 256, 512],
        'h_fea_len': [64, 128, 256, 512],
        'n_conv': [1, 2, 3, 4, 5],
        'n_h': [1, 2, 3, 4, 5]
    }
    best_loss = float('inf')
    best_params = None
    for params in ParameterGrid(param_space):
        loss = run_main(params)
        print(f"Params: {params}, Loss: {loss}")
        if loss < best_loss:
            best_loss = loss
            best_params = params
    print(f"Best Loss: {best_loss}, Best Params: {best_params}")

def bayesian_optimization(trials=10):
    def objective_function(lr, batch_size, weight_decay, atom_fea_len, h_fea_len, n_conv, n_h):
        set_all_seeds(42)
        params = {
            'lr': lr,
            'batch_size': int(round(batch_size)) , # / 16) * 16),
            'weight_decay': weight_decay,                        ######RELATION?
            'atom_fea_len': int(round(atom_fea_len)),
            'h_fea_len': int(round(h_fea_len)),
            'n_conv': int(round(n_conv)),
            'n_h': int(round(n_h))
        }
        return -run_main(params)

    # pbounds = {
    #     'lr': (1e-8, 1e-3),
    #     'batch_size': (16, 64),
    #     'weight_decay': (1e-6, 1e-2),
    #     'atom_fea_len': (64, 512),
    #     'h_fea_len': (64, 512),
    #     'n_conv': (1, 5),
    #     'n_h': (1, 5)
    # }
    pbounds = {
    'atom_fea_len': (64, 512),
    'batch_size': (16, 64),
    'h_fea_len': (64, 512),
    'lr': (1e-5, 1e-3),              # log-scale
    'n_conv': (1, 5),
    'n_h': (1, 5),
    'weight_decay': (1e-5, 1e-2)     # log-scale
    }

    # optimizer = BayesianOptimization(f=objective_function, pbounds=pbounds, random_state=42)
    optimizer = BayesianOptimization(f=objective_function,pbounds=pbounds,random_state=42,verbose=2)    
    optimizer.maximize(init_points=20,n_iter=trials)  
    ##########################################################################################################
    targets = [-res['target'] for res in optimizer.res]
    best_so_far = []
    best = float('inf')
    for t in targets:
        best = min(best, t)
        best_so_far.append(best)

    plt.plot(best_so_far, label='Best so far')
    plt.plot(targets, linestyle='--', label='Raw loss')
    plt.xlabel("Iteration")
    plt.ylabel("Validation Loss")
    plt.title("Bayesian Optimization Progress")
    plt.legend()
    plt.grid()
    plt.show()
    ######################################################################################################
    print(f"Best Loss: {-optimizer.max['target']}, Best Params: {optimizer.max['params']}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['random', 'grid', 'bayesian'], required=True)
    parser.add_argument('--trials', type=int, default=10)
    args = parser.parse_args()

    if args.method == 'random':
        random_search(args.trials)
    elif args.method == 'grid':
        grid_search()
    elif args.method == 'bayesian':
        bayesian_optimization(args.trials)

if __name__ == '__main__':
    main()
