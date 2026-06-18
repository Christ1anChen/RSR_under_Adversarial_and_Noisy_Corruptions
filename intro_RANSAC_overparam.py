# ==============================================================================
# Introduction: compare RANSAC+ with RANSAC under different dimension of subspace r*
# when the input subspace dimension of RANSAC is overparameterized by one
# ==============================================================================
import numpy as np
import matplotlib.pyplot as plt
from RSR_methods import RANSAC, RANSAC_PLUS
from intro_all_methods import generate_adversarial_toy_data, compute_subspace_distance

# ==============================================================================
# CONFIGURATION PARAMETERS
# ==============================================================================
base_seed = 2024
np.random.seed(base_seed)
d = 100       # Ambient dimension
n = 500       # Sample size
eps = 0.2     # Adversarial corruption rate
r_adv = 2     # Rank of adversarial subspace
K = 5         # Number of repetitions

tab_r = [10, 20, 30, 40]
methods = ["RANSAC (r = r* + 1)", "RANSAC+"]

# Metric storage matrices: shape (len(tab_r), K, len(methods))
error_tab = np.zeros((len(tab_r), K, len(methods)))
time_tab = np.zeros((len(tab_r), K, len(methods)))


# ==============================================================================
# MAIN EXPERIMENTAL LOOP
# ==============================================================================
print("=" * 90)
print(f"STARTING SUBSPACE OVERPARAMETERIZATION SWEEP (K={K} TRIALS)")
print("=" * 90)

for idx_r, r in enumerate(tab_r):
    print(f"\nEvaluating Baseline Target Rank r* = {r}")
    
    for trial in range(K):
        current_seed = base_seed + trial
        rng = np.random.default_rng(current_seed)
        
        # Generate (eps, Sig_xi)-corrupted sample set matrix
        X, V_true = generate_adversarial_toy_data(
            d=d, r=r, r_adv=r_adv, N=n, epsilon=eps, seed=current_seed
        )

        # --- Standard RANSAC (Overparameterized by 1) ---
        V_RANSAC, _, t_ransac = RANSAC(X, d=r+1, T_max=1000000, threshold=0.1)
        error_tab[idx_r, trial, 0] = compute_subspace_distance(V_RANSAC, V_true)
        time_tab[idx_r, trial, 0] = t_ransac

        # --- RANSAC+ (Fully Adaptive) ---
        th_nominal = 0.1
        V_ran, r_est, t_ran = RANSAC_PLUS(X, th=th_nominal, st=1.0, eps=eps, T_max=1000000)
        error_tab[idx_r, trial, 1] = compute_subspace_distance(V_ran, V_true)
        time_tab[idx_r, trial, 1] = t_ran
        
        print(f"  Trial {trial+1}/{K} | Rank {r} -> RANSAC Err: {error_tab[idx_r, trial, 0]:.4e} | RANSAC+ Err: {error_tab[idx_r, trial, 1]:.4e} (Est r: {r_est})")

    # Interim calculation metrics reporting
    avg_errs = np.median(error_tab[idx_r], axis=0)
    print(f"--> [Median sin theta] RANSAC (r+1): {avg_errs[0]:.4e} | RANSAC+ (Adaptive): {avg_errs[1]:.4e}")

# Save experimental data arrays safely
np.save("saved_data/error_tab_RANSAC_overparam.npy", error_tab)
np.save("saved_data/time_tab_RANSAC_overparam.npy", time_tab)
print("\n--> Overparameterization benchmarking tracking completed and saved to disk.")

# ==============================================================================
# DATA AGGREGATION & VISUALIZATION
# ==============================================================================
# Prune the upper/lower 20% quantiles to filter out extreme simulation variance loops
lower_quantile = 0.2
upper_quantile = 0.8

mean_errors = np.zeros((len(tab_r), len(methods)))
std_errors = np.zeros((len(tab_r), len(methods)))

for idx_r in range(len(tab_r)):
    for idx_m in range(len(methods)):
        trial_results = error_tab[idx_r, :, idx_m]
        
        # Calculate percentiles dynamically for inner boundary masking
        q_low = np.percentile(trial_results, lower_quantile * 100)
        q_high = np.percentile(trial_results, upper_quantile * 100)
        
        trimmed_trials = trial_results[(trial_results >= q_low) & (trial_results <= q_high)]
        
        mean_errors[idx_r, idx_m] = np.mean(trimmed_trials)
        std_errors[idx_r, idx_m] = np.std(trimmed_trials)

# --- PLOTTING ENVIRONMENT ---
plt.figure(figsize=(10, 6), dpi=100)

markers = ["x", "o"]
line_styles = ["--", "-"]
colors = ["#1f77b4", "#d62728"]

for idx_m, method in enumerate(methods):
    plt.errorbar(
        tab_r, 
        mean_errors[:, idx_m], 
        yerr=std_errors[:, idx_m],
        marker=markers[idx_m], 
        linestyle=line_styles[idx_m],
        color=colors[idx_m],
        linewidth=2.0, 
        markersize=8, 
        capsize=8,          # Length of the error bar cross-caps
        capthick=2.0,       # Thickness of the cross-caps
        elinewidth=2.0,     # Thickness of the vertical bar line
        alpha=0.8,          # Slight opacity to clarify overlapping bars cleanly
        label=method
    )

# Formatting using raw strings to protect LaTeX markup tokens from escaping
plt.xlabel(r"True Subspace Dimension ($r^{\star}$)", fontsize=15, labelpad=10)
plt.ylabel(r"Distance from the True Subspace ($\| \text{P}_{\widehat{S}} - \text{P}_{S^\star} \|$)", fontsize=15, labelpad=10)

plt.yscale("log")
plt.xticks(tab_r)
plt.grid(True, linestyle="--", alpha=0.5, which="both")

plt.legend(fontsize=15, loc="best", frameon=True, shadow=False)
plt.tight_layout()

# Save image
plt.savefig("Pics/intro_RANSAC_overparam.png", dpi=300)
print("--> Evaluation comparative line plot successfully saved as 'Pics/intro_RANSAC_overparam.png'")
plt.show()

# # Intro: compare RANSAC+ with RANSAC under different dimension of subspace $r^{\star}$
# # when the input subspace dimension of RANSAC is overparameterized by one
# #
# from RSR_methods import *

# # Set random seed
# np.random.seed(2024)

# # Parameters
# d = 100  # ambient dimension
# n = 500  # sample size

# # Record error and runtime under different subspace dimension
# O = sp.stats.ortho_group.rvs(d)
# eps = 0.2  # adversarial corruption rate
# tab_r = [10, 20, 30, 40]  # dimension of subspace
# K = 20  # number of repetitions
# error_tab = []  # error table
# time_tab = []  # runtime table

# # Main loop
# for r in tab_r:
#     error_row = []
#     time_row = []
#     # Construct the true covariance
#     U = O[:, :r]
#     Up = O[:, r:]
#     proj_true = U@U.T  # true projection operator
#     D = [1]*r
#     D = np.diag(D)
#     Sig = U@D@U.T
#     for i in range(K):  # repeat for K times
#         error_lis = []
#         time_lis = []
#         # Generate (eps, Sig_xi)-corrupted sample set
#         X = np.random.multivariate_normal(np.zeros(d), Sig, n)
#         Xc, cind = acr(X, Up, eps)
#         # Empirical (PCA)
#         _, S, ceigvecs = np.linalg.svd(Xc, full_matrices=False)
#         ceigvecs = ceigvecs.T
#         error_lis.append(np.linalg.norm(ceigvecs[:, :r]@ceigvecs[:, :r].T - proj_true))
#         # RANSAC
#         V_RANSAC, flag, runtime_ransac = RANSAC(Xc, r+1, tau=0.1, T_max=1e6)
#         error_lis.append(np.linalg.norm(V_RANSAC@V_RANSAC.T - proj_true))
#         time_lis.append(runtime_ransac)
#         # RANSAC+
#         th = 0.01
#         st = 1.
#         V_fp, resid, runtime_fp = First_Phase(Xc, th, st)
#         pXc = Xc@V_fp  # projection
#         rec, ind_tab, eigs_tab, runtime_sp = Second_Phase(pXc, T_min=1e1, T_max=1e6, eps=eps)
#         print(rec[:r+2])
#         res_V, runtime_pp = Post_Process(pXc, V_fp, rec, ind_tab, eigs_tab)
#         error_lis.append(np.linalg.norm(res_V@res_V.T - proj_true))
#         time_lis.append(runtime_fp+runtime_sp+runtime_pp)
#         print(r, i, error_lis, time_lis)
#         # Record error and runtime
#         error_row.append(error_lis)
#         time_row.append(time_lis)
#     # Record error and runtime
#     error_tab.append(error_row)
#     time_tab.append(time_row)
# error_tab = np.asarray(error_tab)
# time_tab = np.asarray(time_tab)

# # Save
# np.save("error_tab_RANSAC_overparam.npy", error_tab)
# np.save("time_tab_RANSAC_overparam.npy", time_tab)