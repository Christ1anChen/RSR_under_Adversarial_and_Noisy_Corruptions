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
K = 10        # Number of repetitions

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
        V_RANSAC, _, t_ransac = RANSAC(X, d=r+1, T_max=10000000, threshold=0.1)
        error_tab[idx_r, trial, 0] = compute_subspace_distance(V_RANSAC, V_true)
        time_tab[idx_r, trial, 0] = t_ransac

        # --- RANSAC+ (Fully Adaptive) ---
        th_nominal = 0.1
        V_ran, r_est, t_ran = RANSAC_PLUS(X, th=th_nominal, st=1.0, eps=eps, T_max=1000000)
        error_tab[idx_r, trial, 1] = compute_subspace_distance(V_ran, V_true)
        time_tab[idx_r, trial, 1] = t_ran
        
        print(f"  Trial {trial+1}/{K} | Rank {r} -> RANSAC Err: {error_tab[idx_r, trial, 0]:.4e} | RANSAC+ Err: {error_tab[idx_r, trial, 1]:.4e} (Est r: {r_est})")

    # Interim calculation metrics reporting
    avg_errs = np.min(error_tab[idx_r], axis=0)
    print(f"--> [Median sin theta] RANSAC (r+1): {avg_errs[0]:.4e} | RANSAC+ (Adaptive): {avg_errs[1]:.4e}")

# Save experimental data arrays safely
np.save("saved_data/error_tab_RANSAC_overparam.npy", error_tab)
np.save("saved_data/time_tab_RANSAC_overparam.npy", time_tab)
print("\n--> Overparameterization benchmarking tracking completed and saved to disk.")

# ==============================================================================
# DATA AGGREGATION & VISUALIZATION
# ==============================================================================
errors = np.zeros((len(tab_r), len(methods)))

for idx_r in range(len(tab_r)):
    for idx_m in range(len(methods)):
        # Extract the K trials for this specific configuration
        trial_results = error_tab[idx_r, :, idx_m]
        
        # Track the absolute lowest subspace distance error recorded
        errors[idx_r, idx_m] = np.min(trial_results)

# --- PLOTTING ENVIRONMENT ---
plt.figure(figsize=(10, 6), dpi=100)

markers = ["x", "o"]
line_styles = ["--", "-"]
colors = ["#1f77b4", "#d62728"]

for idx_m, method in enumerate(methods):
    plt.plot(
        tab_r, 
        errors[:, idx_m], 
        marker=markers[idx_m], 
        linestyle=line_styles[idx_m],
        color=colors[idx_m],
        linewidth=3.0, 
        markersize=10, 
        alpha=0.9,
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

