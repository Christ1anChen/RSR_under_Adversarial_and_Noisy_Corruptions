# ==============================================================================
# Introduction: compare RANSAC+ with RANSAC under different Gaussian noise intensity $\sigma^2$
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
r = 10        # Target subspace dimension
r_adv = 2     # Adversarial contamination rank
eps = 0.2     # Adversarial corruption rate
K = 20        # Number of repetitions

sig2_tab = [1e-4, 1e-3, 1e-2, 1e-1]
methods = ["RANSAC", "RANSAC+"]

# Metric storage matrices: shape (len(sig2_tab), K, len(methods))
error_tab = np.zeros((len(sig2_tab), K, len(methods)))
time_tab = np.zeros((len(sig2_tab), K, len(methods)))


# ==============================================================================
# MAIN EXPERIMENTAL LOOP
# ==============================================================================
print("=" * 90)
print(f"STARTING COMPARATIVE NOISE SWEEP (K={K} TRIALS)")
print("=" * 90)

for idx_sig, sig2 in enumerate(sig2_tab):
    print(f"\nEvaluating Noise Level sigma^2 = {sig2}")

    for trial in range(K):  # repeat for K times
        current_seed = base_seed + trial
        rng = np.random.default_rng(current_seed)
        error_lis = []
        time_lis = []

        # Generate (eps, Sig_xi)-corrupted sample set
        X, V_true = generate_adversarial_toy_data(
            d=d, r=r, r_adv=r_adv, N=n, epsilon=eps, seed=base_seed+trial
            )
        
        # Add zero-mean isotropic Gaussian noise matrix
        if sig2 > 0.0:
            gaussian_noise = rng.normal(0.0, np.sqrt(sig2/d), size=X.shape)   # Covariance matrix is (sig2/d)*I_d
            gn_X = X + gaussian_noise
        else:
            gn_X = X.copy()
        print(f"Trial {trial+1}/{K} for eps={eps:.2f}, sig^2={sig2:.1e} - Data generated with seed {base_seed+trial}")

        # --- Standard RANSAC ---
        V_RANSAC, _, t_ransac = RANSAC(gn_X, d=r, T_max=10000, threshold=0.1)
        error_tab[idx_sig, trial, 0] = compute_subspace_distance(V_RANSAC, V_true)
        time_tab[idx_sig, trial, 0] = t_ransac


        # --- RANSAC+ ---
        th = np.maximum(2.0 * np.sqrt(sig2), 0.1)
        V_ran, _, t_ran = RANSAC_PLUS(gn_X, th, st=1.0, eps=eps, T_max=10000)
        error_tab[idx_sig, trial, 1] = compute_subspace_distance(V_ran, V_true)
        time_tab[idx_sig, trial, 1] = t_ran

    # Log progress summary to console
    avg_errs = np.median(error_tab[idx_sig], axis=0)
    print(f"--> [Median sin theta] RANSAC: {avg_errs[0]:.4e} | RANSAC+: {avg_errs[1]:.4e}")

# Save data
np.save("saved_data/error_tab_RANSAC_noisy.npy", error_tab)
np.save("saved_data/time_tab_RANSAC_noisy.npy", time_tab)


# ==============================================================================
# DATA AGGREGATION & VISUALIZATION
# ==============================================================================
errors = np.zeros((len(sig2_tab), len(methods)))

for idx_sig in range(len(sig2_tab)):
    for idx_m in range(len(methods)):
        # Extract the K trials for this specific configuration
        trial_results = error_tab[idx_sig, :, idx_m]
        
        # Track the median subspace distance error recorded
        errors[idx_sig, idx_m] = np.median(trial_results)

# --- PLOTTING ENVIRONMENT ---
plt.figure(figsize=(10, 6), dpi=100)

markers = ["x", "o"]
line_styles = ["--", "-"]
colors = ["#1f77b4", "#d62728"]

for idx_m, method in enumerate(methods):
    plt.plot(
        sig2_tab, 
        errors[:, idx_m], 
        marker=markers[idx_m], 
        linestyle=line_styles[idx_m],
        color=colors[idx_m],
        linewidth=3.0, 
        markersize=10, 
        alpha=0.9,
        label=method
    )

# Formatting the plot
plt.xlabel(r"Gaussian Noise Intensity ($\sigma^2$)", fontsize=15, labelpad=10)
plt.ylabel(r"Distance from the True Subspace ($\| \text{P}_{\widehat{S}} - \text{P}_{S^\star} \|$)", fontsize=15, labelpad=10)

plt.xscale("log")
plt.yscale("log")
plt.xticks(sig2_tab)
plt.grid(True, linestyle="--", alpha=0.5, which="both")

plt.legend(fontsize=15, loc="best", frameon=True, shadow=False)
plt.tight_layout()

# Save image
plt.savefig("Pics/intro_RANSAC_noisy.png", dpi=300)
print("--> Evaluation line plot saved as 'Pics/intro_RANSAC_noisy.png'")
plt.show()

