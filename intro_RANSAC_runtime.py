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
d = 10000     # Large ambient dimension
n = 500       # Sample size
r_adv = 2     # Rank of adversarial subspace
eps = 0.2     # Adversarial corruption rate
K = 1         # Number of repetitions

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
        current_seed = base_seed + trial
        rng = np.random.default_rng(current_seed)

        # Generate (eps, Sig_xi)-corrupted sample set matrix
        X, V_true = generate_adversarial_toy_data(
            d=d, r=r, r_adv=r_adv, N=n, epsilon=eps, seed=current_seed
        )
        
        # --- Standard RANSAC ---
        T_max = int((1.0 / (1.0 - 2*eps)) ** r)
        # Enforce execution safety clamps on exponential expansions
        T_max = np.clip(T_max, 100, int(1e12))
        max_attempts = int(1e12)
        
        V_RANSAC, max_inliers, rt_ransac = RANSAC(X, d=r, T_max=T_max, threshold=0.1, max_attempts=max_attempts)
    
        error_tab[idx_r, trial, 0] = compute_subspace_distance(V_RANSAC, V_true)
        time_tab[idx_r, trial, 0] = rt_ransac

        # --- RANSAC+ ---
        th_nominal = 0.1
        V_ran, r_est, t_ran = RANSAC_PLUS(X, th=th_nominal, st=1.0, eps=eps, T_min=100, T_max=int(1e12))
        
        error_tab[idx_r, trial, 1] = compute_subspace_distance(V_ran, V_true)
        time_tab[idx_r, trial, 1] = t_ran
        
        print(f"Trial {trial+1:02d} | Rank {r:2d} -> RANSAC Time: {rt_ransac:.4f}s | RANSAC+ Time: {t_ran:.4f}s (Est r: {r_est})")

    # Console summary logging
    avg_times = np.median(time_tab[idx_r], axis=0)
    print(f"--> [Median Runtime] RANSAC: {avg_times[0]:.4f}s | RANSAC+: {avg_times[1]:.4f}s")

# Save matrix data structures safely
np.save("saved_data/error_tab_RANSAC_runtime.npy", error_tab)
np.save("saved_data/time_tab_RANSAC_runtime.npy", time_tab)
print("\n--> Benchmark time evaluation records successfully saved to disk.")


# ==============================================================================
# DATA VISUALIZATION
# ==============================================================================
times = np.zeros((len(tab_r), len(methods)))

for idx_r in range(len(tab_r)):
    for idx_m in range(len(methods)):
        # Extract the K execution times for this specific configuration
        trial_runtimes = time_tab[idx_r, :, idx_m]
        
        # Track the absolute minimum execution time recorded (best-case speed)
        times[idx_r, idx_m] = np.min(trial_runtimes)

# --- PLOTTING GENERATOR ---
plt.figure(figsize=(10, 6), dpi=100)

markers = ["x", "o"]
line_styles = ["--", "-"]
colors = ["#1f77b4", "#d62728"]

for idx_m, method in enumerate(methods):
    if method == "RANSAC":
        plt.plot(
            tab_r[:3], 
            times[:3, idx_m], 
            marker=markers[idx_m], 
            linestyle=line_styles[idx_m],
            color=colors[idx_m],
            linewidth=3.0, 
            markersize=10, 
            alpha=0.9, 
            label=method
        )
    else:
        plt.plot(
            tab_r, 
            times[:, idx_m], 
            marker=markers[idx_m], 
            linestyle=line_styles[idx_m],
            color=colors[idx_m],
            linewidth=3.0, 
            markersize=10, 
            alpha=0.9, 
            label=method
        )

# Clean LaTeX label typesetting
plt.xlabel(r"True Subspace Dimension ($r^{\star}$)", fontsize=15, labelpad=10)
plt.ylabel(r"Runtime (seconds)", fontsize=15, labelpad=10)

plt.yscale("log")  # Using log scale since RANSAC time climbs exponentially while RANSAC+ scales mildly
plt.xticks(tab_r)
plt.grid(True, linestyle="--", alpha=0.5, which="both")
plt.legend(fontsize=15, loc="best", frameon=True, shadow=False)
plt.tight_layout()

# Save performance graphic output
plt.savefig("Pics/intro_RANSAC_runtime.png", dpi=300)
print("--> Empirical complexity scaling chart saved to 'Pics/intro_RANSAC_runtime.png'")
plt.show()

