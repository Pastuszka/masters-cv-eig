import numpy as np
from scipy import sparse
from scipy.sparse.linalg import ArpackNoConvergence
from scipy.stats import norm, binom
from scipy.linalg import interpolative
from numpy import ndarray


def edge_splitting(A, eps: float,
                   is_directed: bool = False):
    """Performs the edge splitting procedure.

    Splits the adjacency matrix into train and test. Each edge from the
    original matrix is put into the test matrix with probability eps. Otherwise
    it is put into the train matrix.

    Args:
        A (sparray): Adjacency matrix as scipy sparse array.
        eps (float): Probability of putting an edge in test matrix.
        is_directed (bool): Should the graph be treated as directed. If false,
            only the upper triangular is considered during splitting, and is
            then mirrored to retain symmetry.

    Returns:
        tuple[sparray, sparray]: Train and test adjacency matrices
    """
    A_to_split = A
    if not is_directed:
        A_to_split =  sparse.dok_array(sparse.triu(A_to_split), dtype=int)
    A_test = sparse.dok_array(A.shape, dtype=int)
    ix, iy = A_to_split.nonzero()
    A_test[ix, iy] = binom.rvs(A_to_split[ix, iy].todense(), eps)
    if not is_directed:
        A_test[iy, ix] = A_test[ix, iy]
    A_train = A - A_test
    return A_train, A_test


def test_stat(A, A_test, x: ndarray, eps) -> float:
    """Computes the test statistic in the CV eigenvalues algorithm.

    Args:
        A (sparray): Full adjacency matrix of the graph.
        A_test (sparray): Test adjacency matrix obtained from edge splitting.
        x (ndarray): Eigenvector of the train adjacency matrix.
        eps (float): splitting probability used in edge splitting.

    Returns:
        float: Value of the test statistic.
    """
    x = x.flatten()
    lam_test = x.T @ A_test @ x
    x2 = x ** 2
    A_diag = A.diagonal()
    sigma = np.sqrt(2 * eps * (x2).T @ A @ x2 - eps * np.sum(x2 * A_diag * x2))
    t = lam_test / sigma
    return t


def norm_reg_matrix(A):
    """Computes the normalized and regularized adjacency matrix.

    Args:
        A (sparray): Adjacency matrix.

    Returns:
        sparray: The normalized and regularized adjacency matrix.
    """
    d = A.sum(1)
    tau = np.mean(d)
    
    if tau == 0:
        return A
    D = sparse.diags(1 / np.sqrt(d + tau))
    L = D @ A @ D
    return L


def gspectral(A_train, kmax):
    try:
        w, v = sparse.linalg.eigs(A_train, k=kmax, which='LR')
    except ArpackNoConvergence as e:
        w = e.eigenvalues
        v = e.eigenvectors
        print(f"Eigs did not converge. Computed {len(w)} eigenvectors")

    #ind = np.argsort(np.abs(w))[::-1]
    #v = v[:, ind]
    return w, v


def gspectral_exact(A_train, kmax):
    w, v = np.linalg.eig(A_train.todense())

    ind = np.argsort(np.abs(w))[::-1]
    v = v[:, ind]
    return w, v


def eig_cv(A, kmax: int, eps: float = 0.2, alpha: float = 0.05,
           folds: int = 5, normalize: bool = True,
           is_directed: bool = False, return_t: bool = False) -> int:
    """Estimates the graph dimension using cross-validated eigenvalues.

    Args:
        A (sparray): Adjacency matrix
        kmax (int): maximal graph dimension to consider
        eps (float): splitting probability for edge splitting
        alpha (float): significance level for the test statistic
        folds (int): number of cv folds to perform
        normalize (bool): should the matrix A be normalized and regularized
        is_directed (bool): should the graph be treated as directed

    Returns:
        int: The computed dimension of the graph.
    """
    t = np.empty((folds, kmax))
    for i in range(folds):
        A_train, A_test = edge_splitting(A, eps, is_directed)
        if normalize:
            A_train = norm_reg_matrix(A_train)

        w, v = gspectral(A_train, kmax)

        for j in range(0, kmax):
            if j >= len(w):
                t[i, j] = np.nan
                continue
            eigenvec = v[:,j].real
            t[i,j] = test_stat(A, A_test, eigenvec, eps)
    t_mean = np.nanmean(t, 0)
    if return_t:
        return t_mean
    p_val = 1 - norm.cdf(t_mean)
    p_val_greater = p_val >= alpha
    if any(p_val_greater):
        k = np.min(np.nonzero(p_val_greater))
    else:
        k = len(p_val_greater)
    return k


def eig_cv_nie_pomaga(A, kmax: int, eps: float = 0.2, alpha: float = 0.05,
           folds: int = 5, normalize: bool = True,
           is_directed: bool = False, return_t: bool = False) -> int:
    """Estimates the graph dimension using cross-validated eigenvalues.

    Args:
        A (sparray): Adjacency matrix
        kmax (int): maximal graph dimension to consider
        eps (float): splitting probability for edge splitting
        alpha (float): significance level for the test statistic
        folds (int): number of cv folds to perform
        normalize (bool): should the matrix A be normalized and regularized
        is_directed (bool): should the graph be treated as directed

    Returns:
        int: The computed dimension of the graph.
    """
    t = np.empty((folds, kmax))
    for i in range(folds):
        A_train, A_test = edge_splitting(A, eps, is_directed)
        if normalize:
            A_train = norm_reg_matrix(A_train)

        w, v = np.linalg.eig(A_train.todense())

        ind = np.argsort(np.abs(w))[::-1]
        v = v[:, ind]
        for j in range(0, kmax):
            if j >= len(w):
                t[i, j] = np.nan
                continue
            eigenvec = np.copy(v[:,j])
            t[i,j] = test_stat(A, A_test, eigenvec, eps)
    t_mean = np.nanmean(t, 0)
    if return_t:
        return t_mean
    p_val = 1 - norm.cdf(t_mean)
    p_val_greater = p_val >= alpha
    if any(p_val_greater):
        k = np.min(np.nonzero(p_val_greater))
    else:
        k = len(p_val_greater)
    return k

def eig_cv_mod(A, kmax: int, eps: float = 0.2,
               normalize: bool = True, is_directed: bool = False) -> int:
    """Estimates the graph dimension using the modifed CV eigenvalues.

    This version of the algortihm is used in some technical proofs in the
    CV eigenvalues paper. Not recommended in real uses.

    Args:
        A (sparray): Adjacency matrix.
        kmax (int): Maximal graph dimension to consider,
        eps (float): Splitting probability for edge splitting,
        normalize (bool): Should the matrix A be normalized and regularized,
        is_directed (bool): Should the graph be treated as directed,

    Returns:
        int: The computed dimension of the graph.
    """
    A_train, A_test = edge_splitting(A, eps, is_directed)
    k = 1

    if normalize:
        A_train = norm_reg_matrix(A_train)
    try:
        w, v = sparse.linalg.eigs(A_train, k=kmax)
    except ArpackNoConvergence as e:
        w = e.eigenvalues
        v = e.eigenvectors
        print(f"Eigs did not converge. Computed {len(w)} eigenvectors")

    ind = np.argsort(w)[::-1]
    v = v[:, ind]
    n = A.shape[0]
    log_n = np.log(n)
    tk_min = np.sqrt(n * log_n)
    for j in range(1, kmax):
        x = v[:, j].real.flatten()
        lam_test = x.T @ A_test @ x
        x2 = x**2
        sigma = np.sqrt((eps/(1-eps)) * (x2.T @ (
            2*A_train - sparse.diags(A_train.diagonal())) @ x2))
        norm_x = np.max(np.abs(x))**2
        max_x = np.min([sigma**2 / log_n**2, log_n / n])
        if norm_x <= max_x:
            tk = lam_test / sigma
            if tk >= tk_min:
                k += 1
    return k


def _find_eigenvectors(H, t, n):
    """Supporting function used in the B-H algorithm"""
    w, v = np.linalg.eig(H)
    v = v.T
    sorted_eig = sorted(zip(w, v), key=lambda x: x[0])[::-1]
    k = n-1
    for i in range(1, n):
        if t * sorted_eig[n - i][0] > sorted_eig[n - i - 1][0]:
            k = i - 1
            break
    if k == 0:
        return np.empty((n, 0))
    smallest = sorted_eig[-k:]
    return np.array([vec for val, vec in smallest]).T


def bethe_hessian_matrix(r: float, A, d: int = None) -> ndarray:
    """Computes the Bethe-Hessian matrix of A.

    Args:
        r (float): The r parameter of the B-H matrix.
        A (sparray): Adjacency matrix.
        d (int, optional): The node dimensions. Computed in not given.

    Returns:
        ndarray: The Bethe-Hessian matrix.
    """
    if d is None:
        d = np.sum(A, axis=1)
    n = len(d)
    I = np.identity(n)
    D = np.diag(d)
    return (r**2 - 1)*I - r*A + D


def bethe_hessian(A, t: float = 5.0, r: float = None) -> int:
    """Estimates the graph dimension using the Bethe-Hessian matrix.

    Args:
        A (sparray): Adjacency matrix.
        t (float): The t parameter in the algorithm. Defaults to 5.0
        r (float, optional): The r parameter in the algorithm. If not provided,
            recommended value is computed.

    Returns:
        int: The computed dimension of the graph.
    """
    d = A.sum(1)
    A = A.todense()
    n = len(d)
    if r is None:
        r = np.sqrt(np.sum(d) / n)

    H_assort = bethe_hessian_matrix(r, A, d)
    H_disassort = bethe_hessian_matrix(-r, A, d)
    assort_points = _find_eigenvectors(H_assort, t, n)
    disassort_points = _find_eigenvectors(H_disassort, t, n)
    points = np.column_stack((assort_points, disassort_points))

    return points.shape[1]


def b_matrix(A):
    """Computes the non-backtracking matrix of A.

    Args:
        A (sparray): Adjacency matrix.

    Returns:
        B (sparray): The Non-backtracking matrix.
    """
    n = A.shape[0]
    d = A.sum(1)
    D = sparse.diags(d)
    I = sparse.identity(n)
    B = sparse.bmat([[None, D-I], [-I, A]])
    return B


def non_backtracking(A, kmax: int, exact: bool = True) -> int:
    """Estimates the graph dimension using the non-backtracking matrix.

    Args:
        A (sparray): Adjacency matrix.
        kmax (int): Maximal graph dimension to consider.
        exact (float): Should the exact algorithm be used to compute the matrix
            norm. Requires the dense form of A, more memory intensive.

    Returns:
        int: The computed dimension of the graph.
    """
    B = b_matrix(A)

    if exact:
        b_n = np.linalg.norm(B.todense(), 2)
    else:
        b_n = interpolative.estimate_spectral_norm(B)
    try:
        w, v = sparse.linalg.eigs(B, k=kmax)
    except ArpackNoConvergence as e:
        w = e.eigenvalues
        v = e.eigenvectors
        print(f"Eigs did not converge. Computed {len(w)} eigenvectors")
    return np.sum(w > np.sqrt(b_n))
