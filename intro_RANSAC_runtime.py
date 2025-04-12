# Intro: compare the runtime of RANSAC+ with RANSAC
# under different subspace dimension $r^{\star}$
# 
from RSR_methods import *

# Set random seed
np.random.seed(2024)

# Parameters
d = 1000  # ambient dimension
n = 500  # sample size

# Record error and runtime under different subspace dimension
O = sp.stats.ortho_group.rvs(d)
df = 3  # degree of freedom of (inlier) student-t distribution
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
        X = gen_student_t(Sig, df, n)
        Xc, cind = acr(X, Up, eps)
        # Empirical (PCA)
        _, S, ceigvecs = np.linalg.svd(Xc, full_matrices=False)
        ceigvecs = ceigvecs.T
        error_lis.append(np.linalg.norm(ceigvecs[:, :r]@ceigvecs[:, :r].T - proj_true))
        # RANSAC
        r_ = 1
        runtime_ransac = 0
        while r_ <= r+1:
            T_max = int((1/(1-2*eps))**r_)
            V_RANSAC, flag, rt_ransac = RANSAC(Xc, r_, tau=0.1, T_max=T_max)
            runtime_ransac = runtime_ransac + rt_ransac
            if flag == 1:
                print("RANSAC algorithm terminates at rank:", r_)
                break
            r_ = r_+1
        error_lis.append(np.linalg.norm(V_RANSAC@V_RANSAC.T - proj_true))
        time_lis.append(runtime_ransac)
        # RANSAC+
        th = 0.1
        st = 1.
        V_fp, resid, runtime_fp = First_Phase(Xc, th, st)
        pXc = Xc@V_fp
        rec, ind_tab, eigs_tab, runtime_sp = Second_Phase(pXc, T_min=2e1, T_max=1e6, eps=eps)
        print(rec[:r+2])
        res_V, runtime_pp = Post_Process(Xc, r, rec, ind_tab, eigs_tab)
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

# save
np.save("error_tab_RANSAC_runtime.npy", error_tab)
np.save("time_tab_RANSAC_runtime.npy", time_tab)
