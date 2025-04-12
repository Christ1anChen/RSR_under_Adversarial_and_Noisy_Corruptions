# Intro: compare RANSAC+ with RANSAC under different dimension of subspace $r^{\star}$
# when the input subspace dimension of RANSAC is overparameterized by one
#
from RSR_methods import *

# Set random seed
np.random.seed(2024)

# Parameters
d = 100  # ambient dimension
n = 500  # sample size

# Record error and runtime under different subspace dimension
O = sp.stats.ortho_group.rvs(d)
eps = 0.2  # adversarial corruption rate
tab_r = [10, 20, 30, 40]  # dimension of subspace
K = 20  # number of repetitions
error_tab = []  # error table
time_tab = []  # runtime table

# Main loop
for r in tab_r:
    error_row = []
    time_row = []
    # Construct the true covariance
    U = O[:, :r]
    Up = O[:, r:]
    proj_true = U@U.T  # true projection operator
    D = [1]*r
    D = np.diag(D)
    Sig = U@D@U.T
    for i in range(K):  # repeat for K times
        error_lis = []
        time_lis = []
        # Generate (eps, Sig_xi)-corrupted sample set
        X = np.random.multivariate_normal(np.zeros(d), Sig, n)
        Xc, cind = acr(X, Up, eps)
        # Empirical (PCA)
        _, S, ceigvecs = np.linalg.svd(Xc, full_matrices=False)
        ceigvecs = ceigvecs.T
        error_lis.append(np.linalg.norm(ceigvecs[:, :r]@ceigvecs[:, :r].T - proj_true))
        # RANSAC
        V_RANSAC, flag, runtime_ransac = RANSAC(Xc, r+1, tau=0.1, T_max=1e6)
        error_lis.append(np.linalg.norm(V_RANSAC@V_RANSAC.T - proj_true))
        time_lis.append(runtime_ransac)
        # RANSAC+
        th = 0.01
        st = 1.
        V_fp, resid, runtime_fp = First_Phase(Xc, th, st)
        pXc = Xc@V_fp  # projection
        rec, ind_tab, eigs_tab, runtime_sp = Second_Phase(pXc, T_min=1e1, T_max=1e6, eps=eps)
        print(rec[:r+2])
        res_V, runtime_pp = Post_Process(pXc, V_fp, rec, ind_tab, eigs_tab)
        error_lis.append(np.linalg.norm(res_V@res_V.T - proj_true))
        time_lis.append(runtime_fp+runtime_sp+runtime_pp)
        print(r, i, error_lis, time_lis)
        # Record error and runtime
        error_row.append(error_lis)
        time_row.append(time_lis)
    # Record error and runtime
    error_tab.append(error_row)
    time_tab.append(time_row)
error_tab = np.asarray(error_tab)
time_tab = np.asarray(time_tab)

# Save
np.save("error_tab_RANSAC_overparam.npy", error_tab)
np.save("time_tab_RANSAC_overparam.npy", time_tab)