# ==============================================================================
# Introduction: Comparison of all methods in the noiseless case
# under different adversarial corruption rate $\epsilon > 0$
# ==============================================================================
from RSR_methods import *
from tabulate import tabulate
import matplotlib.pyplot as plt


# ==============================================================================
# ADVERSARIAL DATA GENERATION FUNCTION
# ==============================================================================
def generate_adversarial_toy_data(d=100, r=10, r_adv=2, N=1000, epsilon=0.2, seed=2025):
    """
    Generates N clean samples from a rank-(r) covariance matrix and replaces an epsilon fraction
    with adversarial outliers from a rank-(r_adv) covariance matrix that is perfectly orthogonal
    to the clean subspace.
    """
    rng = np.random.default_rng(seed)
    
    # Generate a random global orthogonal basis for R^d
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    
    # Isolate the true clean basis (first r columns)
    V_true = Q[:, :r]
    
    # Isolate the adversarial basis from the remaining orthogonal complement
    # This guarantees that the eigenvectors of Sigma_hat are perfectly orthogonal to V_true
    V_adv = Q[:, r : r + r_adv]
    
    # Define eigenvalues (variances) for both distributions
    # Giving outliers a slightly larger variance mimics realistic adversarial corruptions
    eigenvals_clean = rng.uniform(1.0, 1.0, size=r)
    eigenvals_adv = rng.uniform(10.0, 10.0, size=r_adv)
    
    # Calculate sample counts
    N_adv = int(N * epsilon)
    N_clean = N - N_adv
    
    print(f"Generating Data: Total Samples={N} | Clean Samples={N_clean} | Adversarial Outliers={N_adv} (epsilon={epsilon})")
    
    # Generate clean samples: X_clean = V_true * Lambda_clean^(1/2) * Z
    Z_clean = rng.normal(size=(r, N_clean))
    X_clean = V_true @ (np.sqrt(eigenvals_clean)[:, None] * Z_clean)
    
    # Generate adversarial outliers: X_adv = V_adv * Lambda_adv^(1/2) * Z
    Z_adv = rng.normal(size=(r_adv, N_adv))
    X_adv = V_adv @ (np.sqrt(eigenvals_adv)[:, None] * Z_adv)
    
    # Combine together into the contaminated data matrix X
    X = np.hstack([X_clean, X_adv])
    
    # Shuffle columns so the algorithms don't know which samples are the outliers
    shuffled_indices = rng.permutation(N)
    X = X[:, shuffled_indices]
    
    return X, V_true


# ==============================================================================
# METRIC FUNCTION: SUBSPACE DISTANCE
# ==============================================================================
def compute_subspace_distance(V_est, V_true):
    """
    Computes the spectral distance between two subspaces spanned by V_est and V_true.
    Returns sin(theta_max), where theta_max is the largest principal angle.
    0.0 = Perfect match, 1.0 = Completely orthogonal.
    """
    # Handle potentially different rank outputs
    if V_est.shape[1] != V_true.shape[1]:
        # If ranks mismatch, compute distance via projection operator norms
        P_est = V_est @ V_est.T
        P_true = V_true @ V_true.T
        return np.linalg.norm(P_est - P_true, ord=2)
        
    # Standard ultra-fast principal angle calculation
    inner_prod = V_est.T @ V_true
    cosines = np.linalg.svd(inner_prod, compute_uv=False)
    min_cos = np.clip(np.min(cosines), -1.0, 1.0)
    return np.sqrt(1.0 - min_cos**2)


# ==============================================================================
# EXPERIMENT EXECUTION
# ==============================================================================
if __name__ == "__main__":
    print("="*80)
    print("STARTING ROBUST SUBSPACE RECOVERY TOY EXPERIMENT")
    print("="*80)
    
    # Configuration parameters
    d = 100            # Ambient dimension
    r = 10             # True subspace rank
    r_adv = 2          # Adversarial subspace rank
    N = 1000           # Total sample size
    K = 20             # Number of independent repetitions per epsilon configuration
    base_seed = 2025   # Base random seed
    epsilons = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]  # Adversarial contamination fractions to sweep over
    
    # Initialize dictionaries to hold tracking metrics across the sweep
    algs = ["STE", "FMS", "GGD", "RPCA", "RF", "RANSAC", "RANSAC+"]
    error_tracking = {alg: [] for alg in algs}
    time_tracking = {alg: [] for alg in algs}
    
    print("=" * 80)
    print(f"STARTING PARAMETER SWEEP OVER EPSILON: {epsilons} ({K} TRIALS EACH)")
    print("=" * 80)
    
    for eps in epsilons:
        print(f"\n[EVALUATING CONTAMINATION FRACTION: epsilon = {eps}]")

        # Temporary arrays to accumulate trial runs for the current epsilon block
        accumulated_errors = {alg: [] for alg in algs}
        accumulated_times = {alg: [] for alg in algs}
        
        for trial in range(K):
            print(f"\n--> Trial {trial + 1}/{K} for epsilon = {eps}")
            # Dynamic seed iteration
            current_seed = base_seed + trial

            # Generate new data matrix matching the current contamination rate and random seed
            X, V_true = generate_adversarial_toy_data(
                d=d, r=r, r_adv=r_adv, N=N, epsilon=eps, seed=current_seed
            )
            
            # --- Run STE ---
            V_ste, t_ste = STE(X, d=r, T_max=10000, tau=1e-4, gamma=1e-3)
            accumulated_errors["STE"].append(compute_subspace_distance(V_ste, V_true))
            accumulated_times["STE"].append(t_ste)
            
            # --- Run FMS ---
            V_fms, t_fms = FMS(X, d=r, p=1.0, T_max=10000, tau=1e-4, epsilon=1e-10)
            accumulated_errors["FMS"].append(compute_subspace_distance(V_fms, V_true))
            accumulated_times["FMS"].append(t_fms)
            
            # --- Run GGD ---
            V_ggd, t_ggd = GGD(X, d=r, tau=1e-4, K=10, T_max=10000)
            accumulated_errors["GGD"].append(compute_subspace_distance(V_ggd, V_true))
            accumulated_times["GGD"].append(t_ggd)

            # --- Run RPCA ---
            V_rpc, t_rpc = RPCA(X, d=r, T_max=10000, tau=1e-4, chunk_size=500)
            accumulated_errors["RPCA"].append(compute_subspace_distance(V_rpc, V_true))
            accumulated_times["RPCA"].append(t_rpc)
            
            # --- Run RF ---
            V_rf, _, t_rf = RF(X, d=r, num_iters=100000, tol=1e-2)
            accumulated_errors["RF"].append(compute_subspace_distance(V_rf, V_true))
            accumulated_times["RF"].append(t_rf)

            # --- Run Classic RANSAC ---
            V_ransac_base, _, t_ransac_base = RANSAC(X, d=r, T_max=400000, threshold=1e-2, max_attempts=100)
            accumulated_errors["RANSAC"].append(compute_subspace_distance(V_ransac_base, V_true))
            accumulated_times["RANSAC"].append(t_ransac_base)

            # --- Run RANSAC+ ---
            V_ran, _, t_ran = RANSAC_PLUS(X, th=1e-2, st=1.2, eps=eps, T_max=100000)
            accumulated_errors["RANSAC+"].append(compute_subspace_distance(V_ran, V_true))
            accumulated_times["RANSAC+"].append(t_ran)

        # Compute and record the median across the K completed trials
        # print(f"Performance tabulate of the accumulated errors for the current epsilon (eps = {eps}):")
        # print(accumulated_errors)
        for alg in algs:
            error_tracking[alg].append(np.median(accumulated_errors[alg]))
            time_tracking[alg].append(np.median(accumulated_times[alg]))
            
        print(f"--> Completed all {K} trials for epsilon = {eps} successfully.")

    # ==============================================================================
    # GENERATE FINAL SWEEP BENCHMARK REPORTS
    # ==============================================================================
    headers = ["Algorithm Baseline"] + [f"eps = {e}" for e in epsilons]
    
    # Construct Error Rows
    error_rows = []
    for alg in algs:
        row = [alg] + [f"{err:.4e}" for err in error_tracking[alg]]
        error_rows.append(row)
        
    # Construct Time Rows
    time_rows = []
    for alg in algs:
        row = [alg] + [f"{t:.4f}s" for t in time_tracking[alg]]
        time_rows.append(row)
        
    print("\n" + "=" * 90)
    print("FINAL SWEEP REPORT: SUBSPACE RECOVERY ERROR (sin θ)")
    print("=" * 90)
    print(tabulate(error_rows, headers=headers, tablefmt="grid"))
    
    print("\n" + "=" * 90)
    print("FINAL SWEEP REPORT: EXECUTION TIME")
    print("=" * 90)
    print(tabulate(time_rows, headers=headers, tablefmt="grid"))
    print("=" * 90)

    # ==============================================================================
    # PLOTTING
    # ==============================================================================
    plt.figure(figsize=(10, 6), dpi=100)
    
    # Mapping distinct marker structures to separate overlapping baselines
    markers = {
        "STE": 'o', "FMS": 's', "GGD": '^', "RANSAC+": 'D', 
        "RPCA": 'v', "RANSAC": 'x', "RF": 'p'
    }
    line_styles = {
        "STE": '-', "FMS": '--', "GGD": '-.', "RANSAC+": '-', 
        "RPCA": ':', "RANSAC": '--', "RF": '-.'
    }
    
    for alg in algs:
        plt.plot(
            epsilons, 
            error_tracking[alg], 
            marker=markers[alg], 
            linestyle=line_styles[alg],
            linewidth=2.0, 
            markersize=6, 
            label=alg
        )
        
    # Polishing graph layout
    # plt.title("Subspace Recovery Accuracy Under Adversarial Poisoning", fontsize=15, fontweight='bold', pad=15)
    plt.xlabel(r"Adversarial Corruption Parameter ($\epsilon$)", fontsize=12, labelpad=10)
    plt.ylabel(r"Distance from the True Subspace ($\sin \theta_{\max}$)", fontsize=12, labelpad=10)
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(epsilons)
    plt.yscale('log')  # Uses a log scale to cleanly isolate fine precision limits
    
    plt.legend(fontsize=10, loc='best', frameon=True, shadow=False)
    plt.tight_layout()
    
    # Save a clean copy to disk and pop open the window
    plt.savefig("Pics/subspace_error_vs_epsilon.png", dpi=300)
    print("\n--> Plot graphic successfully saved as 'subspace_error_vs_epsilon.png'")
    plt.show()
