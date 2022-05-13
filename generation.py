from argparse import ArgumentError
import numpy as np
from pydantic import NumberNotMultipleError
from scipy import sparse
from scipy.sparse.linalg import ArpackNoConvergence
from scipy.stats import norm, binom
from scipy.linalg import interpolative
from numbers import Number
import sys

def simple_block_connectivity_matrix(nblocks, p_within, p_between):
    B = np.full((nblocks, nblocks), p_between)
    np.fill_diagonal(B, p_within)
    return B


def dc_sbm(theta, z, B, symmetric=True, expected_degree=None):
    n = len(theta)
    theta2 = np.outer(theta, theta)
    B_z = B[np.ix_(z,z)]
    P = theta2 * B_z
    if expected_degree:
        P = P * expected_degree * n / np.sum(P)

    E = binom.rvs(1, P)

    if symmetric:
        E = np.triu(E)
        E = E.T + np.triu(E, 1)
    return E


def bin_tree_distance_matrix(k: int) -> np.ndarray:
    """
    Create a matrix b such that b(a,b) is the distance from the nearest
    common ancestor of leaves a and b to the root in a complete binary tree

    Parameters
    ----------
    k : int
        Number of leaves in the tree. Must be a power of 2.
    """
    n = 1
    max_distance = np.log2(k)
    current_matrix = np.full((n, n), max_distance)
    while n != k:
        max_distance -= 1
        previous_matrix = current_matrix
        current_matrix = np.full((2*n, 2*n), max_distance)
        current_matrix[0:n, 0:n] = previous_matrix
        current_matrix[n:2*n, n:2*n] = previous_matrix
        n *= 2
    return current_matrix


def bin_tree_prob_matrix(
    blocks: np.ndarray, 
    expected_degree: Number) -> np.ndarray:
    """
    Create an probability matrix in hierarchical stochastic block model.
    Blocks are connected hierarchically, in such a way that the probability of
    connection is proportional to 2 to the power of the distance from their 
    nearest common ancestor in a complete binary tree where each block is a
    leaf. Number of blocks is therefore assumed to be the smallest power of 2
    equal or greater to max(blocks)

    Parameters
    ----------
    blocks : array
        Block membership for each node. We assume blocks are numbered from 0.
    expected_degree: Number
        Average expected degree for nodes.
    """
    # find the smallest power of 2 greater or equal than max(blocks)
    k = 2 ** np.ceil(np.log2(np.max(blocks)))

    g = bin_tree_distance_matrix(k)
    b = 2 ** g[np.ix_(blocks, blocks)]
    current_degree = np.mean(np.sum(b, 1))
    b = b * expected_degree / current_degree
    return b


def degree_corrected_sbm(
    theta: np.ndarray, 
    b: np.ndarray) -> np.ndarray:
    """
    Create an adjacency matrix in the degree corrected stochastic block model.
    Blocks are connected hierarchically, in such a way that the probability of
    connection is proportional to 2 to the power of the distance from their 
    nearest common ancestor in a complete binary tree where each block is a
    leaf. Number of blocks is therefore assumed to be the smallest power of 2
    equal or greater to max(blocks)

    Parameters
    ----------
    theta : array
        Degree correction values for each node.
    blocks : array
        Block membership for each node. We assume blocks are numbered from 0
    expected_degree: Number
        Average expected degree for nodes.
    """
    p = np.outer(theta, theta) * b
    E = binom.rvs(1, p)
    return E