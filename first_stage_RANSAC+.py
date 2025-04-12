# First stage: verifying the overestimation of dimension in the first stage of RANSAC+
# 
import numpy as np
from RSR_methods import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

# Variables: corruption rate $\epsilon$ and intensity of Gaussian noise $\sigma^2$
tab_eps = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]  # corruption rate
tab_sig2 = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 0.0]  # Gaussian noise intensity

# Generate clean sample set from Gaussian distribution
X = np.random.multivariate_normal(np.zeros(d), Sig, n)

# Record overestimation of subspace dimension in first stage
K = 20  # Number of repetitions
dim_mat = np.zeros((len(tab_eps), len(tab_sig2), K))
time_mat = np.zeros((len(tab_eps), len(tab_sig2), K))

# Main loop
for i, eps in enumerate(tab_eps):
    for j, sig2 in enumerate(tab_sig2):
        for k in range(K):  # repeat for K times
            # Generate (eps, Sig_xi)-corrupted sample set
            Xc, cind = acr(X, Up, eps)
            gn_Xc = gn(Xc, sig2)
            # First stage of RANSAC+
            th = np.maximum(2*np.sqrt(sig2), 0.1)  # threshold
            st = 1.  # step-size
            V_fp, resid, runtime_fp = First_Phase(gn_Xc, th, st)
            dim_mat[i,j,k] = V_fp.shape[1]
            time_mat[i,j,k] = runtime_fp

# Construct dataframe of dimension overestimation
avg_dim_mat = dim_mat.mean(axis=2)
ratio_dim_mat = (avg_dim_mat/r).T
df = pd.DataFrame(ratio_dim_mat, index=tab_sig2, columns=tab_eps)

# Plot heatmap
plt.figure(figsize = (12,8))
ax = sns.heatmap(df, annot=True, fmt=".2f", cmap="viridis", linewidths=0.2,
                 annot_kws={'size': 15, 'color': 'red'},
                 cbar_kws={'label': "Overestimation of Subspace Dimension $\hat{r} / r^{\star}$"})
cbar = ax.collections[0].colorbar
cbar.set_label(
    "Overestimation of Subspace Dimension $\hat{r} / r^{\star}$", 
    size=20,                # Font size
    weight="normal",        # Font weight (normal, bold, etc.)
    rotation=270,           # Rotation (270 = vertical)
    labelpad=25             # Distance from color bar
)
ax.set_xlabel('Adversarial Corruption Rate $\epsilon$', fontsize=20)
ax.set_ylabel('Gaussian Noise Intensity $\sigma^2$', fontsize=20)
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()