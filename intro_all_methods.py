# Intro: Comparison of all methods in the noiseless case
# under different adversarial corruption rate $\epsilon$
#
from RSR_methods import *

# Set random seed
np.random.seed(2024)

# Parameters
r = 10   # subspace dimension
d = 100  # ambient dimension
n = 500  # sample size

# True covariance
O = np.eye(d)
U = O[:, :r]
Up = O[:, r:]
proj_true = U@U.T  # true projection operator
D = [1]*r
D = np.diag(D)
Sig = U@D@U.T

# Variable: adversarial corruption rate $\epsilon$
tab_eps = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]  # corruption rate
K = 20  # number of repetitions
error_tab = []  # error table
time_tab = []  # runtime table

# Generate clean sample set
X = np.random.multivariate_normal(np.zeros(d), Sig, n)

# Main loop
for eps in tab_eps:
    error_row = []
    time_row = []
    for i in range(K):  # repeat for K times
        # Record error and runtime
        error_lis = []
        time_lis = []
        # Generate eps-corrupted sample set
        Xc, cind = acr(X, Up, eps)
        # Empirical (PCA)
        _, S, ceigvecs = np.linalg.svd(Xc, full_matrices=False)
        ceigvecs = ceigvecs.T
        error_lis.append(np.linalg.norm(ceigvecs[:, :r]@ceigvecs[:, :r].T - proj_true))
        # TME
        eigvecs_tme, runtime_tme = TME(Xc, T_max=1e2)
        V_TME = eigvecs_tme[:, :r]
        error_lis.append(np.linalg.norm(V_TME@V_TME.T - proj_true))
        time_lis.append(runtime_tme)
        # FMS
        V_FMS, runtime_fms = FMS(Xc, L0=ceigvecs[:, :r], T_max=1e2)
        error_lis.append(np.linalg.norm(V_FMS@V_FMS.T - proj_true))
        time_lis.append(runtime_fms)
        # GDD
        V_GDD, runtime_gdd = GDD(Xc, V0=ceigvecs[:, :r], T_max=1e3)
        error_lis.append(np.linalg.norm(V_GDD@V_GDD.T - proj_true))
        time_lis.append(runtime_gdd)
        # HM
        V_HM, runtime_hm = HM(Xc, r, T_max=1e3)
        error_lis.append(np.linalg.norm(V_HM@V_HM.T - proj_true))
        time_lis.append(runtime_hm)
        # RANSAC
        V_RANSAC, flag, runtime_ransac = RANSAC(Xc, r, tau=0.1, T_max=1e6)
        error_lis.append(np.linalg.norm(V_RANSAC@V_RANSAC.T - proj_true))
        time_lis.append(runtime_ransac)
        # RANSAC+
        th = 0.01  # threshold
        st = 1.  # step-size
        V_fp, resid, runtime_fp = First_Phase(Xc, th, st)
        pXc = Xc@V_fp  # projection
        rec, ind_tab, eigs_tab, runtime_sp = Second_Phase(pXc, T_min=5, T_max=1e6, eps=eps)
        print(rec[:r+2])
        res_V, runtime_pp = Post_Process(pXc, V_fp, rec, ind_tab, eigs_tab)
        error_lis.append(np.linalg.norm(res_V@res_V.T - proj_true))
        time_lis.append(runtime_fp+runtime_sp+runtime_pp)
        print(eps, i, error_lis, time_lis)
        # Record error and runtime
        error_row.append(error_lis)
        time_row.append(time_lis)
    # Record error and runtime
    error_tab.append(error_row)
    time_tab.append(time_row)
error_tab = np.asarray(error_tab)
time_tab = np.asarray(time_tab)

# Save
np.save("error_tab_intro_all_methods.npy", error_tab)
np.save("time_tab_intro_all_methods.npy", time_tab)