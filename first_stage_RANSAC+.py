# ==============================================================================
# First Stage: verifying the overestimation of dimension in the first stage of RANSAC+
# ==============================================================================
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from intro_all_methods import generate_adversarial_toy_data, compute_subspace_distance
from RSR_methods import RANSAC_PLUS


# ==============================================================================
# EXPERIMENT PARAMETERS & INITIALIZATION
# ==============================================================================
base_seed = 2025   # Base random seed
r = 10             # True subspace rank (r*)
d = 100            # Ambient dimension (d)
n = 1000           # Sample size (n)
r_adv = 2          # Rank of the adversarial contamination space

# Variables: corruption rate epsilon and intensity of Gaussian noise sigma^2
tab_eps = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.49]
tab_sig2 = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 0.0]

K = 20  # Number of independent repetitions per parameter pair
dim_mat = np.zeros((len(tab_eps), len(tab_sig2), K))
dist_mat = np.zeros((len(tab_eps), len(tab_sig2), K))

print("=" * 90)
print(f"STARTING PHASE 1 DIMENSION SWEEP ({K} TRIALS PER CELL)")
print("=" * 90)


# ==============================================================================
# EXPERIMENT MAIN EXECUTION LOOP
# ==============================================================================
for i, eps in enumerate(tab_eps):
    for j, sig2 in enumerate(tab_sig2):
        for k in range(K):
            # Dynamic seed iteration ensures statistical variability across trials
            current_seed = base_seed + k
            rng = np.random.default_rng(current_seed)
            
            # Generate the (eps)-corrupted underlying low-rank data matrix
            X_gen, V_true = generate_adversarial_toy_data(
                d=d, r=r, r_adv=r_adv, N=n, epsilon=eps, seed=current_seed
            )
            
            # Add zero-mean isotropic Gaussian noise matrix manually
            if sig2 > 0.0:
                gaussian_noise = rng.normal(0.0, np.sqrt(sig2/d), size=X_gen.shape)   # Covariance matrix is (sig2/d)*I_d
                gn_Xc = X_gen + gaussian_noise
            else:
                gn_Xc = X_gen.copy()
            print(f"Trial {k+1}/{K} for eps={eps:.2f}, sig^2={sig2:.1e} - Data generated with seed {current_seed}")
            
            # Execute Phase 1 of RANSAC+
            th = np.clip(2 * np.sqrt(sig2), 0.1, None)   # Auto-calculated distance threshold
            st = 1.0     # Constant step increment size
            
            V_fp, r_fp, runtime_fp = RANSAC_PLUS(gn_Xc, th, st, eps, fp=True)
            
            # Record results
            dim_mat[i, j, k] = r_fp
            dist_mat[i, j, k] = compute_subspace_distance(V_fp, V_true)
            print(f"--> Trial {k+1}/{K} completed: Estimated rank = {r_fp}, Subspace distance = {dist_mat[i, j, k]:.4f}, Runtime = {runtime_fp:.2f} seconds")


# ==============================================================================
# POST-PROCESSING & DATAFRAME GENERATION
# ==============================================================================
avg_dim_mat = dim_mat.mean(axis=2)
avg_dist_mat = dist_mat.mean(axis=2)
ratio_dim_mat = (avg_dim_mat / r).T  # Yields ratio matrix \hat{r} / r*
avg_dist_mat = avg_dist_mat.T
df_dim = pd.DataFrame(ratio_dim_mat, index=tab_sig2, columns=tab_eps)
df_dist = pd.DataFrame(avg_dist_mat, index=tab_sig2, columns=tab_eps)


# ==============================================================================
# HEATMAP VISUALIZATION
# ==============================================================================
plt.figure(figsize=(16, 8), dpi=100)

# Plot heatmap for dimension overestimation ratio
ax = sns.heatmap(
    df_dim, 
    annot=True, 
    fmt=".2f", 
    cmap="viridis", 
    linewidths=0.2,
    annot_kws={'size': 15, 'color': 'red', 'weight': 'bold'},
    cbar_kws={'label': r"Overestimation of Subspace Dimension $\hat{r} / r^{\star}$"}
)

# Polishing the colorbar typography parameters
cbar = ax.collections[0].colorbar
cbar.set_label(
    r"Overestimation Ratio of Subspace Dimension $\hat{r} / r^{\star}$", 
    size=20, 
    weight="normal", 
    rotation=270, 
    labelpad=25
)

ax.set_xlabel(r"Adversarial Corruption Parameter $\epsilon$", fontsize=20, labelpad=12)
ax.set_ylabel(r"Gaussian Noise Intensity $\sigma^2$", fontsize=20, labelpad=12)
plt.xticks(size=15)
plt.yticks(size=15)
# plt.title(r"Phase 1 Empirical Dimension Selection Capacity ($\hat{r}/r^{\star}$)", fontsize=18, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig("Pics/Phase1_Dimension_Overestimation_Heatmap.png", dpi=300)
plt.show()


plt.figure(figsize=(16, 8), dpi=100)

# Plot heatmap for subspace distance
ax = sns.heatmap(
    df_dist, 
    annot=True, 
    fmt=".2f", 
    cmap="viridis", 
    linewidths=0.2,
    annot_kws={'size': 15, 'color': 'blue', 'weight': 'bold'},
    cbar_kws={'label': r"Subspace Distance $\| \text{P}_{\hat{S}} - \text{P}_{S^{\star}}\|$"}
)

# Polishing the colorbar typography parameters
cbar = ax.collections[0].colorbar
cbar.set_label(
    r"Subspace Distance $\| \text{P}_{\hat{S}} - \text{P}_{S^{\star}}\|$", 
    size=20, 
    weight="normal", 
    rotation=270, 
    labelpad=25
)

ax.set_xlabel(r"Adversarial Corruption Parameter $\epsilon$", fontsize=20, labelpad=12)
ax.set_ylabel(r"Gaussian Noise Intensity $\sigma^2$", fontsize=20, labelpad=12)
plt.xticks(size=15)
plt.yticks(size=15)
plt.tight_layout()
plt.savefig("Pics/Phase1_Subspace_Distance_Heatmap.png", dpi=300)
plt.show()