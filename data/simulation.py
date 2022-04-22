import copy
import random
import numpy as np


def run_mc_repeats_(graph, seed_vec, repeat=200, diffusion_limit=25):
    """Run MC multiple times, and compute influenced probability vector
    """

    influ_mat = np.zeros((graph.prob_matrix.shape[0], diffusion_limit))

    # cumulative_sum = np.zeros((graph.adj_matrix.shape[0]))
    for i in range(repeat):
        
        this_mat = run_mc_(graph, seed_vec, diffusion_limit)
        influ_mat += this_mat

        if i >= 1:
            diff_vec = np.abs(influ_mat[:, -1]/(i+1) - (influ_mat[:, -1] - this_mat[:, -1])/i )
            print('mc:{}, mean node error:{:.6f}, total error:{:.6f}'.format(i, np.mean( diff_vec ), np.sum( diff_vec ) ))

    influ_mat /= repeat
    return influ_mat

    
def run_mc_(graph, seed_vec, diffusion_limit=25) -> np.ndarray:
    '''Run MC once
    args:
            graph: SparseGraph format
            seed_vec: multi-hot vector
    return:
            multi-hot vector of activated nodes
    '''
    activated_vec = copy.copy(seed_vec)
    influ_mat = [seed_vec, ]
    last_activated = np.argwhere(seed_vec == 1).flatten().tolist()
    next_activated = []
    diffusion_count = 0

    while len(last_activated) > 0:
        for u in last_activated:
            try:
                u_neighs = graph.get_neighbors(u) # SparseGraph style
            except:
                u_neighs = graph[u] # networkx style
            
            for v in u_neighs:
                if (activated_vec[v] == 0) and random.random() <= graph.prob_matrix[u, v]: # activated
                    activated_vec[v] = 1
                    next_activated.append(v)
        
        current_activated = copy.copy(activated_vec)
        influ_mat.append(current_activated)

        last_activated = next_activated
        next_activated = []
        diffusion_count += 1
        if len(influ_mat) >= diffusion_limit:
            break
    
    if len(influ_mat) < diffusion_limit:
        for i in range(diffusion_limit - len(influ_mat)):
            influ_mat.append(copy.copy(influ_mat[-1]))

    influ_mat = np.array(influ_mat).T
    return influ_mat

    