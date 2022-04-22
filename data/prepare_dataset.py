import random
import numpy as np
import scipy.sparse as sp
import copy
import pickle
import multiprocessing as mp 
from multiprocessing import Pool, TimeoutError
from pathlib import Path

from sparsegraph import SparseGraph
from iocode import load_dataset
from simulation import run_mc_repeats_

data_dir = Path(__file__).parent

def add_prob_mat(graph: SparseGraph,):
    """Add a diffusion probability matrix to the graph.

    """
    # probability matrix
    prob_data = copy.copy(graph.adj_matrix.data)
    prob_indices = copy.copy(graph.adj_matrix.indices)
    prob_indptr = copy.copy(graph.adj_matrix.indptr)
    prob_shape = copy.copy(graph.adj_matrix.shape)

    for i, v in enumerate(prob_data):
        p = random.choice([0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3], ) # candadite probability values
        prob_data[i] = p
    prob_matrix = sp.csr_matrix((prob_data, prob_indices, prob_indptr), shape=prob_shape)
    graph.prob_matrix = prob_matrix

    return graph

def gen_seed_vec(N, n):
    seed_vec = np.zeros((N,))
    seeds = np.random.randint(0, N, size=n)
    seed_vec[seeds] = 1
    return seed_vec


def add_mc_data(graph: SparseGraph, seed_size_list, num_per_size):
    """Add mc influence results / a matrix list
    influ_mat_list
        A list of influence matrix. Each influence matrix's first column is seed vector, and k-th(k!=-1) columns should be k-th step diffusion influence vector.
        The -1 column (last column) is the final influence vector.
        Currently each influence matrix only have 2 columns: [seed_vec, final influ_vec] 
    """
    N = graph.prob_matrix.shape[0]

    # queue = mp.Queue() # RuntimeError: Queue objects should only be shared between processes through inheritance
    with Pool(processes=30) as pool:
        reses = []
        for seed_size in seed_size_list:
            for i in range(num_per_size):
                seed_vec = gen_seed_vec(N, seed_size)
                # run MCs, multi-processing
                res = pool.apply_async(run_mc_repeats_, (graph, seed_vec, 500))
                reses.append(res)
        pool.close()
        pool.join()
           
    influ_mat_list = [res.get() for res in reses ]
    
    # add influenced prob vectors
    graph.influ_mat_list = np.array(influ_mat_list)

    return graph


def add_new_attr_mat(graph: SparseGraph):
    """Add newly constructed feature matrix to the graph, named graph.newattr_matrix, without altering the original graph.attr_matrix.
    Parameters
    ----------
    graph 
        the SparseGraph graph
    """
    prob_matrix = graph.prob_matrix.toarray().T # tranaspose means in degree
    degree = np.sum(prob_matrix, axis=0) # degree info
    newattr_matrix = np.concatenate((degree.reshape(-1,1), graph.seed_vec.reshape(-1,1)), axis=1) # seed info
    graph.newattr_matrix = newattr_matrix
    return graph

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
args = parser.parse_args()


if __name__ == "__main__":
    """Add prob_matrix, seed_vec, seed_size, influ_mat_list, newattr_matrix from the ORIGINAL dataset.
    """
    graph_name = args.dataset # 'cora_ml', 'citeseer', 'ms_academic', 'pubmed'
    suffix = '.npz'
    new_suffix = '_25c.SG'
    seed_size = 150
    seed_sizes = [50, 100, 150, 200, 250, 300]
    num_per_size = 100

    if ( data_dir / (graph_name + new_suffix) ).exists():
        print('file already exists.')
        exit(0)
    graph = load_dataset(graph_name + suffix)
    graph = graph.standardize(select_lcc=True)

    new_graph = add_prob_mat(graph)
    new_graph = add_mc_data(graph, seed_sizes, num_per_size)
    
    with open(data_dir/(graph_name + new_suffix), 'wb') as f:
        pickle.dump(new_graph, f)
    print('Dumped!')



