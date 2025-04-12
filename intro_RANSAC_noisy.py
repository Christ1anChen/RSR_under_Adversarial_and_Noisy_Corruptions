# Intro: compare RANSAC+ with RANSAC under different Gaussian noise intensity $\sigma^2$
# 
from RSR_methods import *

# Set random seed
np.random.seed(2024)

# Parameters
d = 100  # ambient dimension
n = 500  # sample size
r = 10  # dimension of subspace

# True covariance
O = sp.stats.ortho_group.rvs(d)
U = O[:, :r]
Up = O[:, r:]
proj_true = U@U.T  # true projection operator
D = [1]*r
D = np.diag(D)
Sig = U@D@U.T

# Record error and runtime under different noise levels
sig2_tab = [1e-4, 1e-3, 1e-2, 1e-1]
eps = 0.2  # adversarial corruption rate
K = 20  # number of repetitions
error_tab = []  # error table
time_tab = []  # runtime table

# Main loop
for sig2 in sig2_tab:
    error_row = []
    time_row = []
    for i in range(K):  # repeat for K times
        error_lis = []
        time_lis = []
        # Generate (eps, Sig_xi)-corrupted sample set
        X = np.random.multivariate_normal(np.zeros(d), Sig, n)
        Xc, cind = acr(X, Up, eps)
        Xc = gn(Xc, sig_2=sig2)
        # Empirical (PCA)
        _, S, ceigvecs = np.linalg.svd(Xc, full_matrices=False)
        ceigvecs = ceigvecs.T
        error_lis.append(np.linalg.norm(ceigvecs[:, :r]@ceigvecs[:, :r].T - proj_true))
        # RANSAC
        V_RANSAC, flag, runtime_ransac = RANSAC(Xc, r, tau=0.1, T_max=1e6)
        error_lis.append(np.linalg.norm(V_RANSAC@V_RANSAC.T - proj_true))
        time_lis.append(runtime_ransac)
        # RANSAC+
        th = 0.1
        st = 1.
        V_fp, resid, runtime_fp = First_Phase(Xc, th, st)
        pXc = Xc@V_fp  # projection
        rec, ind_tab, eigs_tab, runtime_sp = Second_Phase(pXc, T_min=1e1, T_max=1e6, eps=eps)
        print(rec[:r+2])
        res_V, runtime_pp = Post_Process(pXc, V_fp, rec, ind_tab, eigs_tab)
        error_lis.append(np.linalg.norm(res_V@res_V.T - proj_true))
        time_lis.append(runtime_fp+runtime_sp+runtime_pp)
        print(sig2, i, error_lis, time_lis)
        # Record error and runtime
        error_row.append(error_lis)
        time_row.append(time_lis)
    # Record error and runtime
    error_tab.append(error_row)
    time_tab.append(time_row)
error_tab = np.asarray(error_tab)
time_tab = np.asarray(time_tab)

# Save
np.save("error_tab_RANSAC_noisy.npy", error_tab)
np.save("time_tab_RANSAC_noisy.npy", time_tab)