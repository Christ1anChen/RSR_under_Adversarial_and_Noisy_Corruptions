# ==============================================================================
# Introduction: compare the runtime of RANSAC+ with 
# RANSAC under different subspace dimension r*
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
d = 1000      # Large ambient dimension
n = 500       # Sample size
df = 3        # Degree of freedom of (inlier) student-t distribution
eps = 0.2     # Adversarial corruption rate
K = 20        # Number of repetitions

tab_r = [10, 20, 30, 40]
methods = ["RANSAC", "RANSAC+"]

# Metric storage matrices: shape (len(tab_r), K, len(methods))
error_tab = np.zeros((len(tab_r), K, len(methods)))
time_tab = np.zeros((len(tab_r), K, len(methods)))


# ==============================================================================
# MAIN EXPERIMENTAL RUNTIME SWEEP LOOP
# ==============================================================================
print("=" * 90)
print(f"STARTING RUNTIME COMPLEXITY SWEEP OVER SUBSPACE RANK (K={K} TRIALS)")
print("=" * 90)

for idx_r, r in enumerate(tab_r):
    print(f"\nEvaluating Computational Complexity for Target Rank r* = {r}")
    
    for trial in range(K):
        # Generate (eps, Sig_xi)-corrupted sample set matrix
        X, V_true = generate_adversarial_toy_data(
            d=d, r=r, r_adv=r_adv, N=n, epsilon=eps, seed=current_seed
        )
        
        # --- Sequential Rank-Climbing Standard RANSAC ---
        r_step = 1
        runtime_ransac_total = 0.0
        V_RANSAC = None
        
        while r_step <= r + 1:
            T_max = int((1.0 / (1.0 - 2.0 * eps)) ** r_step)
            # Enforce execution safety clamps on exponential expansions
            T_max = np.clip(T_max, 10, int(1e6))
            
            V_RANSAC, max_inliers, rt_ransac = RANSAC(X, d=r_step, T_max=T_max, threshold=0.1)
            runtime_ransac_total += rt_ransac
            
            if max_inliers >= n*(1-eps):
                break
            r_step += 1
            
        error_tab[idx_r, trial, 0] = compute_subspace_distance(V_RANSAC, V_true)
        time_tab[idx_r, trial, 0] = runtime_ransac_total

        # --- RANSAC+ ---
        th_nominal = 0.1
        V_ran, r_est, t_ran = RANSAC_PLUS(X_corrupted, th=th_nominal, st=1.0, eps=eps, T_max=5000)
        
        error_tab[idx_r, trial, 1] = compute_subspace_distance(V_ran, U)
        time_tab[idx_r, trial, 1] = t_ran
        
        print(f"  Trial {trial+1:02d} | Rank {r:2d} -> RANSAC Time: {runtime_ransac_total:.4f}s | RANSAC+ Time: {t_ran:.4f}s (Est r: {r_est})")

    # Console summary logging
    avg_times = np.median(time_tab[idx_r], axis=0)
    print(f"--> [Median Runtime] RANSAC: {avg_times[0]:.4f}s | RANSAC+: {avg_times[1]:.4f}s")

# Save matrix data structures safely
np.save("saved_data/error_tab_RANSAC_runtime.npy", error_tab)
np.save("saved_data/time_tab_RANSAC_runtime.npy", time_tab)
print("\n--> Benchmark time evaluation records successfully saved to disk.")

# ==============================================================================
# DATA AGGREGATION & VISUALIZATION (TRIMMED RUNTIME MEANS)
# ==============================================================================
lower_quantile = 0.2
upper_quantile = 0.8

mean_times = np.zeros((len(tab_r), len(methods)))
std_times = np.zeros((len(tab_r), len(methods)))

for idx_r in range(len(tab_r)):
    for idx_m in range(len(methods)):
        # Target execution times for trimmed analysis
        trial_runtimes = time_tab[idx_r, :, idx_m]
        
        q_low = np.percentile(trial_runtimes, lower_quantile * 100)
        q_high = np.percentile(trial_runtimes, upper_quantile * 100)
        
        trimmed_runtimes = trial_runtimes[(trial_runtimes >= q_low) & (trial_runtimes <= q_high)]
        
        mean_times[idx_r, idx_m] = np.mean(trimmed_runtimes)
        std_times[idx_r, idx_m] = np.std(trimmed_runtimes)

# --- PLOTTING GENERATOR ---
plt.figure(figsize=(10, 6), dpi=100)

markers = ["x", "o"]
line_styles = ["--", "-"]
colors = ["#1f77b4", "#d62728"]

for idx_m, method in enumerate(methods):
    plt.errorbar(
        tab_r, 
        mean_times[:, idx_m], 
        yerr=std_times[:, idx_m],
        marker=markers[idx_m], 
        linestyle=line_styles[idx_m],
        color=colors[idx_m],
        linewidth=2.0, 
        markersize=8, 
        capsize=8, 
        capthick=2.0, 
        elinewidth=2.0, 
        alpha=0.85, 
        label=method
    )

# Clean LaTeX label typesetting
plt.xlabel(r"True Subspace Dimension ($r^{\star}$)", fontsize=14, labelpad=10)
plt.ylabel(r"Trimmed Algorithmic Execution Time (seconds)", fontsize=14, labelpad=10)
plt.title(r"Computational Complexity vs. Target Subspace Dimensionality Scaling", fontsize=14, fontweight="bold", pad=15)

plt.yscale("log")  # Using log scale since RANSAC time climbs exponentially while RANSAC+ scales mildly
plt.xticks(tab_r)
plt.grid(True, linestyle="--", alpha=0.5, which="both")

plt.text(0.02, 0.02, f"Note: Displays trimmed inner {int((upper_quantile-lower_quantile)*100)}% quantile runtime distributions.", 
         transform=plt.gca().transAxes, fontsize=10, style="italic", bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

plt.legend(fontsize=13, loc="upper left", frameon=True, shadow=False)
plt.tight_layout()

# Save performance graphic output
plt.savefig("Pics/intro_RANSAC_runtime.png", dpi=300)
print("--> Empirical complexity scaling chart saved to 'Pics/intro_RANSAC_runtime.png'")
plt.show()


# # Intro: compare the runtime of RANSAC+ with RANSAC
# # under different subspace dimension $r^{\star}$
# # 
# from RSR_methods import *

# # Set random seed
# np.random.seed(2024)

# # Parameters
# d = 1000  # ambient dimension
# n = 500  # sample size

# # Record error and runtime under different subspace dimension
# O = sp.stats.ortho_group.rvs(d)
# df = 3  # degree of freedom of (inlier) student-t distribution
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
#         X = gen_student_t(Sig, df, n)
#         Xc, cind = acr(X, Up, eps)
#         # Empirical (PCA)
#         _, S, ceigvecs = np.linalg.svd(Xc, full_matrices=False)
#         ceigvecs = ceigvecs.T
#         error_lis.append(np.linalg.norm(ceigvecs[:, :r]@ceigvecs[:, :r].T - proj_true))
#         # RANSAC
#         r_ = 1
#         runtime_ransac = 0
#         while r_ <= r+1:
#             T_max = int((1/(1-2*eps))**r_)
#             V_RANSAC, flag, rt_ransac = RANSAC(Xc, r_, tau=0.1, T_max=T_max)
#             runtime_ransac = runtime_ransac + rt_ransac
#             if flag == 1:
#                 print("RANSAC algorithm terminates at rank:", r_)
#                 break
#             r_ = r_+1
#         error_lis.append(np.linalg.norm(V_RANSAC@V_RANSAC.T - proj_true))
#         time_lis.append(runtime_ransac)
#         # RANSAC+
#         th = 0.1
#         st = 1.
#         V_fp, resid, runtime_fp = First_Phase(Xc, th, st)
#         pXc = Xc@V_fp
#         rec, ind_tab, eigs_tab, runtime_sp = Second_Phase(pXc, T_min=2e1, T_max=1e6, eps=eps)
#         print(rec[:r+2])
#         res_V, runtime_pp = Post_Process(Xc, r, rec, ind_tab, eigs_tab)
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

# # save
# np.save("error_tab_RANSAC_runtime.npy", error_tab)
# np.save("time_tab_RANSAC_runtime.npy", time_tab)
