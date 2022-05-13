import numpy as np
from scipy import sparse
from scipy.sparse.linalg import ArpackNoConvergence
from scipy.stats import norm, binom
from scipy.linalg import interpolative


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


def edge_splitting(A, eps, is_directed=False):
    A_full = A
    if not is_directed:
        A = sparse.dok_array(sparse.triu(A))
    A_train = sparse.dok_array(A.shape)
    ix, iy = A.nonzero()
    for x, y in zip(ix, iy):
        A_train[x, y] = binom.rvs(A[x, y], eps)

        if not is_directed and x != y:
            A_train[y, x] = A_train[x, y]
    
    A_test = A_full - A_train
    return A_train, A_test


def test_stat(A, A_test, x, eps):
    x = x.flatten()
    lam_test = x.T @ A_test @ x
    x2 = x**2
    A_diag = sparse.diags(A.diagonal())
    sigma = np.sqrt(2*eps*(x2).T @ A @ x2 - eps*(x2).T @ A_diag @ x2)
    return lam_test / sigma


def norm_reg_matrix(A):
    tau = A.sum() / A.shape[0]
    if tau == 0:
        return A
    D = sparse.diags((A.diagonal() + tau) ** (-1/2))
    return D @ A @ D


def eig_cv(A, kmax, eps=0.2, alpha=0.05,
           folds=5, normalize=True, is_directed=False):
    t = np.empty((folds, kmax-1))

    for i in range(folds):
        A_train, A_test = edge_splitting(A, eps, is_directed)
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
        for j in range(1, kmax):
            eigenvec = v[:,j].real
            t[i,j-1] = test_stat(A, A_test, eigenvec, eps)
    t_mean = t.mean(0)
    p_val = 1 - norm.cdf(t_mean)
    p_val_greater = p_val >= alpha
    if any(p_val_greater):
        infer = np.min(np.nonzero(p_val_greater))
    else:
        infer = len(p_val_greater)
    return infer + 1


def eig_cv_mod(A, kmax, eps=0.2, normalize=True, is_directed=False):
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


def find_eigenvectors(H, t, n):
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


def bethe_hessian_matrix(r, A, n, d):
    if d is None:
        d = np.sum(A, axis=1)
    if n is None:
        n = len(d)
    I = np.identity(n)
    D = np.diag(d)
    return (r**2 - 1)*I - r*A + D


def bethe_hessian(A, t=5, r=None):
    d = A.sum(1)
    A = A.todense()
    n = len(d)
    if r is None:
        r = np.sqrt(np.sum(d) / n)

    H_assort = bethe_hessian_matrix(r, A, n, d)
    H_disassort = bethe_hessian_matrix(-r, A, n, d)
    assort_points = find_eigenvectors(H_assort, t, n)
    disassort_points = find_eigenvectors(H_disassort, t, n)
    points = np.column_stack((assort_points, disassort_points))

    return points.shape[1]


def b_matrix(A):
    n = A.shape[0]
    d = A.sum(1)
    D = sparse.diags(d)
    I = sparse.identity(n)
    B = sparse.bmat([[None, D-I], [-I, A]])
    return B


def non_backtracking(A, kmax, exact=True):
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
    # ind = np.argsort(w)[::-1]
    # v = v[:,ind]
    return np.sum(w > np.sqrt(b_n))

