import numpy as np
import scipy as sp
from time import perf_counter

# Intersection of two arrays (based on the first array)
#
def intersection(array_1, array_2):
    return np.array([e for e in array_1 if e in array_2])

# Generate student-t samples with mean 0
# Covariance: Sig; Degree of freedom: df; Sample size: n
#
def gen_student_t(Sig, df, n):
    d = Sig.shape[0]
    z1 = np.random.chisquare(df, n) / df
    z2 = np.random.multivariate_normal(np.zeros(d), Sig, size=n)
    X = z2/np.sqrt(z1)[:, None]/np.sqrt(df/(df-2))
    return X

# Different adversarial corruption rate (fixed corruption pattern)
#
def acr(X, Up, eps=0.1):
    n = X.shape[0]
    cind = np.random.choice(n, int(eps*n), replace=False)
    Xc = X.copy()
    Xc[cind] = np.random.multivariate_normal(np.zeros(2), np.eye(2)*10, size=len(cind))@Up[:, :2].T
    return Xc, cind

# Apply gaussian noise on all samples
#
def gn(Xc, sig_2=1e-4):
    n = Xc.shape[0]
    d = Xc.shape[1]
    gn = np.random.standard_normal(size=(n,d))*np.sqrt(sig_2/d)
    gn_Xc = Xc + gn
    return gn_Xc

# Tyler's M-estimator (TME)
#
def TME(Xc, T_max):
    t_start = perf_counter()
    n = Xc.shape[0]  # sample size
    d = Xc.shape[1]  # dimension
    Sig_old = np.eye(d)/d
    Sig_old_inv = np.eye(d)*d  # initialization
    itr = 0
    while itr < T_max:
        Sig_new = 0
        for i in range(n):
            xc = Xc[i]
            wt = xc@Sig_old_inv@xc
            Sig_new = Sig_new + np.outer(xc, xc) / wt
        Sig_new = Sig_new / np.trace(Sig_new)
        progress = np.linalg.norm(Sig_new-Sig_old)
        if progress < 1e-4:
            print(itr, "Tyler's M-estimator terminates", progress)
            break
        Sig_old = Sig_new
        eigvals_tme, eigvecs_tme = sp.linalg.eigh(Sig_new)
        eigvals_tme = eigvals_tme[::-1]
        eigvecs_tme = eigvecs_tme[:, ::-1]
        eigvals_tme_inv = [1/x if x > 1e-10 else 0 for x in eigvals_tme]
        Sig_old_inv = eigvecs_tme@np.diag(eigvals_tme_inv)@eigvecs_tme.T
        itr = itr + 1
    t_stop = perf_counter()
    if itr == T_max:
        print(itr, "Tyler's M-estimator reaches maximum # iterations", progress)
    return eigvecs_tme, t_stop - t_start

# Fast Median Subspace (FMS)
#
def FMS(Xc, L0, T_max):
    t_start = perf_counter()
    n = Xc.shape[0]  # sample size
    d = Xc.shape[1]  # dimension
    k = L0.shape[1]  # dimension of subspace
    L = L0.copy()  # initial subspace
    proj_L_old = L@L.T
    orth_L = np.eye(d) - proj_L_old
    itr = 0
    while itr < T_max:
        Yc = []
        for i in range(n):
            xc = Xc[i]
            dist_xc = np.linalg.norm(orth_L@xc)
            yc = xc / np.maximum(dist_xc**0.5, 1e-10)
            Yc.append(yc)
        Yc = np.asarray(Yc)
        _, S, Vh = sp.linalg.svd(Yc, full_matrices=False)
        L = Vh[:k].T
        proj_L = L@L.T
        progress = np.linalg.norm(proj_L - proj_L_old)
        if progress < 1e-4:
            print(itr, "Fast Median Subspace method terminates", progress)
            break
        proj_L_old = proj_L
        orth_L = np.eye(d) - proj_L
        itr = itr + 1
    t_stop = perf_counter()
    if itr == T_max:
        print(itr, "Fast Median Subspace method reaches maximum # iterations", progress)
    return L, t_stop - t_start

# theta_1: arccos(sigma_d(P1*P2))
# computing angle between two subspaces
#
def theta_1(V1, V2, k):
    proj_V1 = V1@V1.T
    proj_V2 = V2@V2.T
    L, S, Rh = sp.linalg.svd(proj_V1@proj_V2)
    return np.arccos(np.minimum(S[k-1], 1))

# Geodesic Gradient Descent (GDD)
# subdifferential: - \sum x x^T V / ||Q_V x||
# gradient: Q_v * subdifferential
# 
def GDD(Xc, V0, T_max):
    t_start = perf_counter()
    n = Xc.shape[0]  # sample size
    d = Xc.shape[1]  # dimension
    k = V0.shape[1]  # dimension of subspace
    V = V0.copy()  # initial subspace
    proj_V = V@V.T
    orth_V = np.eye(d) - proj_V
    s0 = 1/d  # initial stepsize
    K = 20  # period
    itr = 0
    while itr < T_max:
        subdiff = 0
        for i in range(n):
            xc = Xc[i]
            orth_xc = orth_V@xc
            resid_xc = np.linalg.norm(orth_xc)
            if resid_xc < 1e-10:
                continue
            subdiff = subdiff + np.outer(xc, V.T@xc) / resid_xc
        grad = orth_V@subdiff
        U1, S, Wh = sp.linalg.svd(grad, full_matrices=False)
        s = s0 / 2**(int(itr/K))  # update stepsize
        V_new = V@(Wh.T@(np.diag(np.cos(S*s))@Wh)) + U1@(np.diag(np.sin(S*s))@Wh)
        itr = itr + 1
        t1 = theta_1(V_new, V, k)  # angle between two consecutive subspaces
        if t1 < 1e-4:  # stop criterion
            print(itr, "Geodesic Gradient Descent terminates", t1)
            break
        V = V_new
        proj_V = V@V.T
        orth_V = np.eye(d) - proj_V
    t_stop = perf_counter()
    if itr == T_max:
        print(itr, "Geodesic Gradient Descent reaches maximum # iterations", t1)
    return V, t_stop - t_start

# RANSAC (Hardt and Moitra 2013)
def HM(Xc, r, T_max=1e3):
    t_start = perf_counter()
    n = Xc.shape[0]
    d = Xc.shape[1]
    itr = 0
    while itr < T_max:
        bind = np.random.choice(n, d, replace=False)
        Xb = Xc[bind]
        rank_Xb = np.linalg.matrix_rank(Xb, tol=1e-4)
        if rank_Xb < d:
            U, S, V = sp.linalg.svd(Xb)
            u = U[:, -1]
            ind_ = []
            for i in range(d):
                if np.abs(u[i]) > 1e-4:
                    ind_.append(i)
            ind = bind[ind_]
            break
        itr += 1
    Xb = Xc[ind]
    U, S, V = sp.linalg.svd(Xb)
    print(itr, "Hardt and Moitra terminates")
    t_stop = perf_counter()
    return V[:r].T, t_stop - t_start
            

# Randomly generate a subset that spans a r-dim subspace
#
def gen_subspace(Xc, r):
    n = Xc.shape[0]
    perm_ind = np.random.permutation(n)
    Xb = Xc[perm_ind[:r]]
    rank_Xb = np.linalg.matrix_rank(Xb)
    i = r+1
    while rank_Xb < r:
        Xb = Xc[perm_ind[:i]]
        rank_Xb = np.linalg.matrix_rank(Xb, tol=1e-4)  # set the tol for matrix rank manually
        i = i+1
    gen_ind = np.sort(perm_ind[:i-1])
    return Xc[gen_ind], gen_ind

# RANSAC (Maunu and Lerman 2019)
#
def RANSAC(Xc, r, tau=0.2, T_max=1e3):
    t_start = perf_counter()
    n = Xc.shape[0]
    m = int(n/2)
    k = 0
    itr = 0
    norm_lis = np.linalg.norm(Xc, axis=1)
    while itr < T_max:
        Xb, bind = gen_subspace(Xc, r)
        _, beigvals, beigvecs = sp.linalg.svd(Xb, full_matrices=False)
        V1 = beigvecs[:r].T  # only pick the top-r eigenvectors
        dist_lis = np.linalg.norm(Xc@V1, axis=1)
        angle_lis = np.arccos(np.minimum(dist_lis/norm_lis, 1))
        c = (angle_lis <= tau).sum()
        itr = itr+1
        if c >= k:
            V = V1
            k = c
        if k > m:
            break
    print("r="+str(r), "itr="+str(itr), "RANSAC algorithm terminates with "+str(np.round(k/n*100, decimals=1))+"% samples")
    if itr == T_max:
        flag = 0
    else:
        flag = 1
    t_stop = perf_counter()
    return V, flag, t_stop - t_start

# First phase
# th: threshold for stopping criterion
# st: step size of search batch size
# 
def First_Phase(Xc, th=0.1, st=2.0):
    t_start = perf_counter()
    n = Xc.shape[0]  # sample size
    d = Xc.shape[1]  # dimension
    B = 2  # initial search batch size
    resid = []
    flag = 0
    while B <= d:
        bind = np.random.choice(n, B, replace=False)
        Xb = Xc[bind]
        _, beigvals, beigvecs = sp.linalg.svd(Xb, full_matrices=False)
        V = beigvecs[:B].T
        res = []
        for i in range(n):
            x = Xc[i]
            res.append(np.linalg.norm(x - V@(V.T@x)))
        resid.append(res)
        med = np.median(res)
        if med < th:  # stop criterion
            if flag < 2:
                flag = flag + 1
                continue
            else:
                break
        else:
            flag = 0
        B = int(B*st)+1  # update search batch size
    print("size of search batch:", B)
    resid = np.asarray(resid)
    t_stop = perf_counter()
    return V, resid, t_stop - t_start

# Second phase
# T_min: minimum number of batches
# T_max: maximum number of batches
#
def Second_Phase(pXc, T_min, T_max, eps=0.1):
    t_start = perf_counter()
    n = pXc.shape[0]
    r_ = pXc.shape[1]
    ind_tab = []
    eigs_tab = []
    rec = 1e10*np.ones(r_)  # record the minimum eigenvalues across all batches
    B = int(2*r_ / (1-eps))  # aggressive choice for batch size
    itr = 0
    while itr < np.maximum(T_min, np.minimum((1/(1-1.5*eps))**B, T_max)):
        batch_ind = np.random.choice(n, B, replace=False)
        ind_tab.append(batch_ind)
        bpXc = pXc[batch_ind]
        _, vals, vecs = sp.linalg.svd(bpXc, full_matrices=False)
        vals = vals**2 / B
        eigs_tab.append(vals)
        rec = np.minimum(rec, vals[:r_])
        itr += 1
    print("number of batches:", itr)
    ind_tab = np.asarray(ind_tab)
    eigs_tab = np.array(eigs_tab)
    t_stop = perf_counter()
    return rec, ind_tab, eigs_tab, t_stop - t_start

# After recovering the approximate low rank r,
# choose the batch with the smallest (r+1)-th eigenvalue
#
def Post_Process(pXc, V, rec, ind_tab, eigs_tab):
    t_start = perf_counter()
    batch_ind = []
    decay_lis = []
    # Identify the largest eigengap
    for i in range(1, len(rec)):
        decay_lis.append(rec[i]/rec[i-1])
        if rec[i] < 1e-10:
            break
    r = np.argmin(decay_lis) + 1
    print("Determined rank:", r)

    # Choose the batch with the smallest (r+1)-th eigenvalue
    js = eigs_tab[:, r].argmin()
    batch_ind = ind_tab[js]
    uniq_batch_ind = np.sort(batch_ind)
    bpXc = pXc[uniq_batch_ind]
    _, vals, vecs = sp.linalg.svd(bpXc, full_matrices=False)
    sp_vecs = vecs[:r].T
    res_vecs = V@sp_vecs
    t_stop = perf_counter()
    return res_vecs, t_stop - t_start
