import numpy as np
import scipy.linalg as sp_linalg
from scipy.sparse.linalg import svds, LinearOperator
from time import perf_counter


# RANSAC+ (Chen, Ma & Fattahi 2025)
# 
def RANSAC_PLUS(X, th=1.0, st=2.0, eps=0.25, T_min=100, T_max=int(1e6), fp=False):
    """
    Robustly identifies a low-dimensional subspace from adversarial/noisy data.
    Automatically determines the intrinsic rank of the subspace.
    Based on Chen, Ma & Fattahi, 2025: "RANSAC Revisited: An Improved Algorithm for Robust Subspace Recovery under Adversarial and Noisy Corruptions"
    
    Parameters:
    - X: (D, N) numpy array (D: ambient dimension, N: number of samples).
    - th: float, threshold for median residual stopping criterion.
    - st: float, step multiplier for expanding batch size.
    - eps: float, contamination fraction parameter.
    - T_min, T_max: int, lower and upper bounds for the number of randomized sampling iterations.
    - fp: bool, if True, only executes the first phase of RANSAC+ for dimension estimation.
    
    Returns:
    - res_vecs: (D, r) numpy array, estimated orthogonal basis of the target subspace.
    - r: int, the estimated rank of the subspace.
    - run_time: float, total execution time in seconds.
    """
    t_start = perf_counter()
    D, N = X.shape

    # Initialize a high-performance random generator
    rng = np.random.default_rng()

    # --- Data-Driven Threshold Calculation ---
    if th is None:
        # Draw up to 200 random samples
        sample_size = min(N, 200)
        sample_cols = rng.choice(N, size=sample_size, replace=False)
        
        # Random sampling
        X_sample = X[:, sample_cols] 
        
        # Compute the global median and MAD safely on the small sample
        X_median = np.median(X_sample, axis=1, keepdims=True)
        abs_deviations = np.abs(X_sample - X_median)
        mad = np.median(abs_deviations, axis=1)
        sigma_array = (1.4826) * mad
        th = np.sqrt(np.sum(sigma_array**2))
        print(f"--> Auto-calculated sparse soft-threshold (via {sample_size} samples): {th:.6f}")
    else:
        print(f"--> Using manual sparse soft-threshold: {th}")
    
    # Pre-compute column norms for vectorized distance calculations
    X_sq_norms = np.linalg.norm(X, axis=0)**2

    # PHASE 1: Coarse-Grained Subspace Search
    B, flag = 2, 0
    V = None
    while B <= D:
        bind = rng.choice(N, B, replace=False)
        # QR decomposition of (D, B) matrix. V gives the column space.
        V, _ = sp_linalg.qr(X[:, bind], mode='economic')
        
        pX_curr = V.T @ X
        # Vectorized projection distances using pre-computed norms
        proj_norms = np.sum(pX_curr**2, axis=0)
        res = np.sqrt(np.maximum(X_sq_norms - proj_norms, 0))
        
        if np.median(res) < th:
            flag += 1
            if flag >= 1:
                break
        else:
            flag = 0
            
        B = np.maximum(int(B * st), B + 1)
    
    if fp:
        run_time = perf_counter() - t_start
        print(f"RANSAC+ Phase 1 completed in {run_time:.4f} seconds with coarse dimension r'={V.shape[1]}")
        return V, V.shape[1], run_time

    # PHASE 2: Fine-Grained Subspace Refinement
    pX = V.T @ X 
    r_ = pX.shape[0]
    print(f"RANSAC+ Phase 1 completed with coarse dimension r'={r_}")
    
    B_sec = int(r_ / (1 - 2*eps))  # 2 * r_ / (1 - eps)
    print(f"RANSAC+ Phase 2 Batch Size (B): {B_sec}")
    
    T_target = int((1 / (1 - eps))**B_sec)  # 1.1 * 
    T = np.clip(T_target, T_min, T_max)
    print(f"RANSAC+ Phase 2 Total Number of Batches (T): {T}")
    
    rec = np.full(r_, np.inf)
    eigs_tab = np.zeros((T, r_))

    # Pre-allocate random indices all at once outside the loop to avoid calling rng.choice T times
    ind_tab = np.array([rng.choice(N, B_sec, replace=False) for _ in range(T)])

    for t in range(T):
        batch_ind = ind_tab[t]
        X_sub = pX[:, batch_ind]
        
        # Compute the small (r_ x r_) Gram matrix
        Gram = (X_sub @ X_sub.T) / B_sec
        
        # Compute eigenvalues directly on the small Gram matrix for efficiency
        raw_vals = np.linalg.eigvalsh(Gram)[::-1] 
        eigs = np.maximum(raw_vals, 0.0)  # Ensure non-negativity
        
        eigs_truncated = eigs[:r_]
        eigs_tab[t, :len(eigs_truncated)] = eigs_truncated
        rec = np.minimum(rec, eigs_truncated)
        
    # PHASE 3: Rank Determination & Lifting
    # denom = np.where(rec[:-1] > 1e-12, rec[:-1], 1e-12)
    # decay_ratios = rec[1:] / denom
    print("Eigenvalues of the best recorded:", rec)


    # Condition: The eigenvalue must drop below your adaptive noise threshold
    th2 = max(th, 1e-4)  # Ensure a reasonable lower bound on the threshold  # / np.sqrt(D)
    condition = (rec[1:] < th2)

    if np.any(condition):
        # Find the FIRST index where the energy drops below the noise floor
        # +1 maps the 0-based index of rec[1:] back to the actual rank dimension
        r = int(np.where(condition)[0][0] + 1)
        print(f"--> Noise threshold reached. Stopping rank selection at r = {r}")
    else:
        # If every single eigenvalue is large, the entire coarse space is full of signal energy
        print("--> All dimensions contain high energy. Setting rank to full coarse dimension.")
        r = int(r_)

    # Determine the target index for evaluating the best batch matrix
    eval_idx = r

    # # Condition: The eigenvalue after the drop must be less than the threshold
    # condition = (rec[1:] < th)

    # if np.any(condition):
    #     # Filter out ratios that don't land below the noise threshold
    #     restricted_ratios = np.where(condition, decay_ratios, np.inf)
    #     r = int(np.argmin(restricted_ratios) + 1)
    # else:
    #     # If every single eigenvalue is large, no noise floor was captured; the rank is full
    #     print("--> All dimensions contain high energy. Setting rank to full coarse dimension.")
    #     r = int(r_)

    # # Determine the target index for the (r+1)-th eigenvalue
    # eval_idx = r
    
    # Boundary Protection: If estimated rank equals the total available dimensions,
    # there is no (r+1)-th eigenvalue. We must fall back to the last available component.
    if eval_idx >= r_:
        eval_idx_safe = r_ - 1
        print(f"--> Rank r={r} is full. No (r+1)-th eigenvalue exists. Falling back to maximizing the r-th component.")
        best_batch_idx = eigs_tab[:, eval_idx_safe].argmax()
    else:
        # Pick the batch that MINIMIZES the (r+1)-th eigenvalue (the noise floor)
        best_batch_idx = eigs_tab[:, eval_idx].argmin()
    
    best_batch_ind = np.sort(ind_tab[best_batch_idx])

    # Extract the optimal low-rank basis
    U_best, _, _ = sp_linalg.svd(pX[:, best_batch_ind], full_matrices=False)
    U_seed = U_best[:, :r]  # Shape: (r_, r)
    
    # Pre-compute the squared norms of the columns inside the projected space pX
    pX_sq_norms = np.linalg.norm(pX, axis=0)**2
    
    # Vectorized distance calculation of all N samples to U_seed within R^{r_}
    proj_sq_norms = np.sum((U_seed.T @ pX)**2, axis=0)
    projected_dists = np.sqrt(np.maximum(pX_sq_norms - proj_sq_norms, 0))
    
    # Aggregate all sample indices bounded by the threshold 'th2'
    aggregated_indices = np.where(projected_dists <= th2/np.sqrt(D))[0]
    
    # Extract the refined basis from the pooled projected consensus set
    if len(aggregated_indices) >= r:
        X_consensus = X[:, aggregated_indices]
        print(fr"--> Projected consensus pooled {len(aggregated_indices)} / {N} samples.")
        
        # Using fast Truncated SVD to strictly pull the top-r ambient components
        try:
            # Setting ncv to give the Arnoldi engine ample search room
            ncv_val = int(np.clip(4 * r, r + 2, min(D - 1, X_consensus.shape[1] - 1)))
            res_vecs, S_final, _ = svds(X_consensus, k=r, ncv=ncv_val, tol=1e-5)
             
            # Sort descending so column 0 is the absolute dominant principal component
            sort_idx = np.argsort(S_final)[::-1]
            res_vecs = res_vecs[:, sort_idx]
            
        except Exception as e:
            print(f"--> Truncated SVD failed ({e}). Falling back to safe lifted basis.")
            res_vecs = V @ U_seed
    else:
        print("--> Warning: Projected consensus set too small. Defaulting to seed batch.")
        res_vecs = V @ U_seed
    
    # # Lift back to ambient space R^D
    # res_vecs = V @ U_best[:, :r]
    
    run_time = perf_counter() - t_start 
    print(f"RANSAC+ completed in {run_time:.4f} seconds with estimated rank {r}")

    return res_vecs, r, run_time


# Robust PCA via Factored Alternating Projections (Fixed-Rank PCP)
#
def RPCA(X, d, th=None, T_max=20, tau=1e-4, chunk_size=500):
    """
    Robust PCA via Factored Alternating Projections (Fixed-Rank PCP)
    Based on Netrapalli, Niranjan, Sanghavi, Anandkumar & Jain, NIPS 2014: "Provable Non-convex Robust PCA"
    
    Parameters:
    - X: (D, N) numpy array (D: ambient dimension, N: number of samples).
    - d: int, target rank of the low-rank background subspace.
    - th: float, soft-thresholding parameter. If None, auto-calculated.
    - T_max: int, maximum number of alternating iterations.
    - tau: float, convergence tolerance based on subspace rotation.
    - chunk_size: int, number of frames processed at once to prevent RAM spikes.
    
    Returns:
    - U: (D, d) numpy array, the robust orthogonal basis for the low-rank subspace.
    - run_time: float, total execution time in seconds.
    """
    t_start = perf_counter()
    D, N = X.shape
    
    # --- Data-Driven Threshold Calculation ---
    if th is None:
        # Initialize a high-performance random generator
        rng = np.random.default_rng()
        
        # Sample up to 200 random frames uniformly across the video
        sample_size = min(N, 200)
        sample_cols = rng.choice(N, size=sample_size, replace=False)
        
        # Random sampling
        X_sample = X[:, sample_cols] 
        
        # Compute the global median and MAD safely on the small sample
        global_median = np.median(X_sample)
        mad_entry = np.median(np.abs(X_sample - global_median))
        
        # Classic Candès et al. heuristic scaled by our robust entry spread
        th = (1.4826 * mad_entry) / np.sqrt(max(D, N))
        print(f"--> Auto-calculated sparse soft-threshold (via {sample_size} samples): {th:.6f}")
    else:
        print(f"--> Using manual sparse soft-threshold: {th}")

    # ==========================================
    # INITIALIZATION (Single Dense Allocation)
    # ==========================================
    # Allocate exactly ONE extra DxN matrix to hold the sparse foreground
    S = np.zeros((D, N), dtype=X.dtype)
    
    # Initialize the low-rank subspace basis using standard Truncated SVD
    U, S_init, _ = svds(X, k=d)
    sort_idx = np.argsort(S_init)[::-1]
    U = U[:, sort_idx]
    
    # ==========================================
    # ALTERNATING OPTIMIZATION LOOP
    # ==========================================
    for k in range(T_max):
        U_old = U.copy()
        
        # STEP 1: Compute the Low-Rank Subspace from (X - S) Implicitly
        def matvec(v):
            return X @ v - S @ v
        def rmatvec(u):
            return X.T @ u - S.T @ u
        def matmat(V_mat):
            return X @ V_mat - S @ V_mat
        def rmatmat(U_mat):
            return X.T @ U_mat - S.T @ U_mat

        X_minus_S_op = LinearOperator(
            shape=(D, N), 
            matvec=matvec, 
            rmatvec=rmatvec, 
            matmat=matmat, 
            rmatmat=rmatmat
        )
        
        # Update the orthogonal basis matrix
        U_new, S_vals, _ = svds(X_minus_S_op, k=d)
        sort_idx = np.argsort(S_vals)[::-1]
        U = U_new[:, sort_idx]
        
        # STEP 2: Compute Low-Rank Projection Coefficients (Tiny d x N matrix)
        # H = U.T @ (X - S)
        H = U.T @ X - U.T @ S
        
        # STEP 3: In-Place Chunked Soft-Thresholding for Sparse Component
        # We process frames in horizontal chunks to avoid expanding a full D x N matrix
        for i in range(0, N, chunk_size):
            end_i = min(i + chunk_size, N)
            
            # Generate the low-rank background chunk on the fly
            L_chunk = U @ H[:, i:end_i]
            
            # Compute the residual chunk in-place inside S
            S[:, i:end_i] = X[:, i:end_i] - L_chunk
            
            # Apply element-wise Soft-Thresholding (Shrinkage) directly to S
            S_chunk = S[:, i:end_i]
            S[:, i:end_i] = np.sign(S_chunk) * np.maximum(np.abs(S_chunk) - th, 0)

        # ==========================================
        # CONVERGENCE CHECK (Principal Angles)
        # ==========================================
        inner_prod = U.T @ U_old
        cosines = np.linalg.svd(inner_prod, compute_uv=False)
        min_cos = np.clip(np.min(cosines), -1.0, 1.0)
        subspace_dist = np.sqrt(1.0 - min_cos**2)
        
        if subspace_dist <= tau:
            # print(f"Robust PCA converged at iteration {k+1}")
            break
            
    run_time = perf_counter() - t_start
    print(f"Robust PCA finished in {k+1} iterations, time taken: {run_time:.4f} seconds")
    
    return U, run_time



# Subspace-Constrained Tyler's M-estimator (STE)
#
def STE(X, d, T_max=100, tau=1e-4, gamma=1e-3):
    """
    Subspace-Constrained Tyler's Estimator (STE)
    Based on Yu, Zhang & Lerman, CVPR 2024: "A Subspace-Constrained Tyler's Estimator and its Applications"
    
    Parameters:
    - X: (D, N) numpy array (D: ambient dimension, N: number of samples).
    - d: int, intrinsic dimension of the target subspace.
    - T_max: int, maximum number of iterations.
    - tau: float, convergence tolerance.
    - gamma: float, suppression parameter for the trailing eigenvalues.
    
    Returns:
    - L: (D, d) numpy array, the basis for the d-dimensional subspace.
    - run_time: float, time taken for convergence.
    """
    t_start = perf_counter()
    D, N = X.shape

    # Pre-compute squared Euclidean norms of the data for extreme speed
    X_sq_norms = np.linalg.norm(X, axis=0)**2

    # Initialize the implicit Shape Matrix parameters
    # Instead of a DxD matrix, we only track the top d eigenvectors and eigenvalues, 
    # plus a scalar 'c' for the isotropic tail.
    c = 1.0 / D
    Lambda_d = np.full(d, 1.0 / D)
    
    # Initialize subspace U using truncated SVD
    U_d, S_init, _ = svds(X, k=d)
    U_d = U_d[:, np.argsort(S_init)[::-1]]
    
    for k in range(T_max):
        U_d_old = U_d.copy()

        # Clipping values at 1e-15 ensures they remain valid floating-point 
        Lambda_d = np.clip(Lambda_d, 1e-15, None)
        c = max(c, 1e-15)
        
        # ==========================================
        # Compute denominators via Woodbury Identity
        # ==========================================
        # Project X onto the current subspace
        P = U_d.T @ X  # Shape: (d, N)
        
        # Calculate the eigenvalue differences for the inversion
        diag_diff = (1.0 / Lambda_d) - (1.0 / c)
        
        # Denominator = (1/c)*||x||^2 + sum( diag_diff * P^2 )
        denominators = (1.0 / c) * X_sq_norms + np.sum(diag_diff[:, None] * (P**2), axis=0)
        denominators = np.clip(denominators, 1e-12, None)
        
        # ==========================================
        # THE GHOST MATRIX: Implicit X_weighted
        # ==========================================
        weights = 1.0 / denominators
        sqrt_w = np.sqrt(weights)
        
        # Define matrix-vector/matrix-matrix operations to mimic X_weighted = X * sqrt_w
        def matvec(v):
            return X @ (sqrt_w * v)
        def rmatvec(u):
            return sqrt_w * (X.T @ u)
        def matmat(V_mat):
            return X @ (sqrt_w[:, None] * V_mat)
        def rmatmat(U_mat):
            return sqrt_w[:, None] * (X.T @ U_mat)

        X_weighted_op = LinearOperator(
            shape=(D, N), 
            matvec=matvec, 
            rmatvec=rmatvec, 
            matmat=matmat, 
            rmatmat=rmatmat
        )

        # ==========================================
        # Form Implicit Z matrix and extract top components
        # ==========================================
        # Fast truncated SVD on the DxN matrix completely avoids creating the DxD Z matrix
        try:
            U_full, S_full, _ = svds(
                X_weighted_op, 
                k=d, 
                ncv=max(4 * d, 40), 
                tol=1e-5, 
                maxiter=2000
            )
        except Exception as e:
            # Fallback emergency tracking for ARPACK convergence issues
            print(f"--> ARPACK convergence warning at iteration {k+1}: {e}. Halting optimization safely.")
            break
        
        # Extract top d eigenvectors and eigenvalues
        sort_idx = np.argsort(S_full)[::-1]
        S_full = S_full[sort_idx]
        U_new = U_full[:, sort_idx]
        vals_d = S_full[:d]**2
        
        # Calculate the trace of Z mathematically to find the tail average
        trace_Z = np.sum(weights * X_sq_norms)
        sigma_tail_avg = max((trace_Z - np.sum(vals_d)) / (D - d), 1e-12)
        
        # ==========================================
        # Suppress tail and Normalize
        # ==========================================
        Lambda_new = vals_d
        c_new = gamma * sigma_tail_avg
        
        trace_new = np.sum(Lambda_new) + (D - d) * c_new
        Lambda_d = Lambda_new / trace_new
        c = c_new / trace_new
        U_d = U_new
        
        # ==========================================
        # Memory-Safe Convergence Check
        # ==========================================
        # Fast Principal Angle calculation instead of DxD Frobenius norm
        # Measures the physical rotation of the subspace
        overlap = np.clip(np.sum((U_d.T @ U_d_old)**2), 0, d)
        subspace_diff = np.sqrt(np.abs(d - overlap))
        
        if subspace_diff < tau:
            # print(f"STE converged at iteration {k+1}")
            break

    run_time = perf_counter() - t_start
    print(f"STE finished in {k+1} iterations, time taken: {run_time:.4f} seconds")

    return U_d, run_time
    


# Fast Median Subspace (FMS_p)
#
def FMS(X, d, p=1.0, T_max=100, tau=1e-4, epsilon=1e-10):
    """
    Fast Median Subspace (FMS_p)
    Based on Lerman & Maunu, I&I 2018: "Fast, Robust and Non-convex Subspace Recovery"
    
    Parameters:
    - X: (D, N) numpy array (D: ambient dimension, N: number of samples).
    - d: int, desired rank of the subspace.
    - p: float, robustness power (0 < p < 2; default: 1.0).
    - T_max: int, maximum number of iterations.
    - tau: float, convergence tolerance for subspace distance.
    - epsilon: float, small constant to prevent division by zero.
    
    Returns:
    - U: (D, d) numpy array, the basis for the d-dimensional subspace.
    - run_time: float, time taken for convergence.
    """
    t_start = perf_counter()
    D, N = X.shape
    
    # Pre-compute squared norms of the data columns for fast distance calculation
    # ||x_i||^2
    X_sq_norms = np.linalg.norm(X, axis=0)**2
    
    # Initialize L0 = L1 via truncated PCA d-subspace
    U, _, _ = svds(X, k=d)
    
    for k in range(T_max):
        # Vectorized Distance & Weighting
        # Calculate the projection squared norms: || U^T x_i ||^2
        proj_sq_norms = np.sum((U.T @ X)**2, axis=0)
        
        # Subspace distance using the Pythagorean theorem: sqrt(||x_i||^2 - ||U^T x_i||^2)
        # Using np.maximum to handle microscopic floating-point negatives
        dists = np.sqrt(np.maximum(X_sq_norms - proj_sq_norms, 0))
        
        # Calculate the denominator scaling factor: max(dist(x_i, L_k)^{(2-p)/2}, epsilon)
        exponent = (2.0 - p) / 2.0
        scale_factors = np.maximum(dists**exponent, epsilon)
        
        # Construct Y by scaling the columns of X
        # Broadcasting automatically divides each column i by scale_factors[i]
        weights = 1.0 / scale_factors
        # Y = X * weights  # Shape: (D, N)

        # ==========================================
        # Fix MEMORY ISSUE: THE GHOST MATRIX
        # ==========================================
        # We define how Y interacts with vectors (v) and matrices (V)
        # mathematically: Y * v = X * (weights * v)
        def matvec(v):
            return X @ (weights * v)

        def rmatvec(u):
            return weights * (X.T @ u)
            
        def matmat(V):
            return X @ (weights[:, None] * V)
            
        def rmatmat(U_mat):
            return weights[:, None] * (X.T @ U_mat)

        # Wrap the rules into an object that svds can read
        Y_op = LinearOperator(
            shape=(D, N), 
            matvec=matvec, 
            rmatvec=rmatvec, 
            matmat=matmat, 
            rmatmat=rmatmat
        )
        
        # Subspace Update
        # Truncated PCA on Y/Y_op to get the new subspace L_{k+1}
        U_new, _, _ = svds(Y_op, k=d)
        
        # Convergence Check
        # Measure dist(L_k, L_{k-1}) using the spectral norm of the projection difference
        inner_prod = U_new.T @ U
        cosines = np.linalg.svd(inner_prod, compute_uv=False)
        min_cos = np.clip(np.min(cosines), -1.0, 1.0) # Clip to prevent floating-point errors
        subspace_dist = np.sqrt(1.0 - min_cos**2)
        
        U = U_new
        
        if subspace_dist <= tau:
            # print(f"FMS (p={p}) converged at iteration {k+1}")
            break
    
    run_time = perf_counter() - t_start
    print(f"FMS (p={p}) finished in {k+1} iterations, time taken: {run_time:.4f} seconds")

    return U, run_time


# Geodesic Gradient Descent (GGD)
#
def GGD(X, d, s=1.0, tau=1e-4, K=10, T_max=100):
    """
    Robust Subspace Recovery by Geodesic Gradient Descent (GGD)
    Based on Maunu, Zhang & Lerman, JMLR 2019: "A Well-Tempered Landscape for Non-convex Robust Subspace Recovery"
    
    Parameters:
    - X: (D, N) numpy array, the data matrix (D: ambient dimension, N: samples).
    - d: int, intrinsic dimension of the target subspace.
    - s: float, initial step-size.
    - tau: float, tolerance for the principal angle convergence criterion.
    - K: int, constant step interval length for step-size decay.
    - T_max: int, maximum number of iterations.
    
    Returns:
    - V: (D, d) numpy array, an orthogonal basis spanning the robust subspace.
    - run_time: float, time taken for convergence.
    """
    t_start = perf_counter()
    D, N = X.shape

    # Pre-compute squared norms of the data columns for fast distance calculation
    # ||x_i||^2
    X_sq_norms = np.linalg.norm(X, axis=0)**2
    
    # Initialization using Truncated PCA
    V, _, _ = svds(X, k=d)
    
    # Loop setup (s is handled dynamically inside the loop)
    for k in range(1, T_max + 1):
        # Compute negative Riemannian gradient
        # Calculate V^T X and the orthogonal projection X_perp
        VtX = V.T @ X               # Shape: (d, N)
        
        # Calculate Euclidean distances || X - V @ VtX ||_2 for each column
        proj_sq_norms = np.sum(VtX**2, axis=0)
        dists = np.sqrt(np.maximum(X_sq_norms - proj_sq_norms, 0))
        dists = np.maximum(dists, 1e-10) # Safeguard against division by zero
        
        # Form the gradient: X_perp * diag(1/dists) * (V^T X)^T
        weights = 1.0 / dists
        Y = weights * VtX           # Shape: (d, N)

        # Calculate: minus_grad = X @ Y.T - V @ (VtX @ Y.T)
        term1 = X @ Y.T             # Shape: (D, d)
        term2 = V @ (VtX @ Y.T)     # Shape: (D, d)
        minus_grad = term1 - term2  # Shape: (D, d)
        
        # Compute the SVD of the negative gradient
        U_k, Sigma_k, Wt_k = sp_linalg.svd(minus_grad, full_matrices=False)
        W_k = Wt_k.T
        
        # Geodesic Exponential Map
        # Step-size decay schedule
        s_k = s / (2 ** (k // K))
        
        # Construct diagonal trigonometric matrices
        cos_S = np.diag(np.cos(Sigma_k * s_k))
        sin_S = np.diag(np.sin(Sigma_k * s_k))
        
        # Geodesic update formula
        V_new = V @ W_k @ cos_S @ Wt_k + U_k @ sin_S @ Wt_k
        
        # Convergence Check (Principal Angle)
        # Measure sin(theta_1) using the spectral norm of the difference in projections
        sin_theta = np.linalg.norm(V_new - V @ (V.T @ V_new), ord=2)
        theta_1 = np.arcsin(np.clip(sin_theta, 0.0, 1.0))
        
        if k > 1 and theta_1 <= tau:
            print(f"GGD converged at iteration {k}")
            V = V_new
            break
            
        V = V_new
    
    run_time = perf_counter() - t_start
    print(f"GGD finished in {k} iterations, time taken: {run_time:.4f} seconds")

    return V, run_time


# Classic RANSAC for Robust Subspace Recovery
#
def RANSAC(X, d, T_max=1000, threshold=0.1, max_attempts=1000):
    """
    RANSAC for Robust Subspace Recovery
    based on Maunu & Lerman, 2019: "Robust Subspace Recovery with Adversarial Outliers"
    
    Parameters:
    - X: (D, N) array, data matrix (columns are data points).
    - d: int, intrinsic dimension of the subspace.
    - T_max: int, maximum number of random sampling iterations.
    - threshold: float, distance threshold for a point to be considered an inlier.
    - max_attempts: int, failsafe to prevent infinite loops on highly degenerate data.
    
    Returns:
    - best_V: (D, d) array, orthogonal basis of the estimated subspace.
    - max_inliers: int, number of inliers in the largest consensus set.
    - run_time: float, time taken for the RANSAC procedure.
    """
    t_start = perf_counter()
    D, N = X.shape
    best_V = None
    max_inliers = -1
    
    valid_iters = 0
    attempts = 0
    
    # Pre-compute X squared norms once for the ultra-fast distance calculation
    X_sq_norms = np.linalg.norm(X, axis=0)**2

    rng = np.random.default_rng()
    
    while valid_iters < T_max and attempts < max_attempts:
        attempts += 1
        
        # Randomly sample exactly d points
        idx = rng.choice(N, d, replace=False)
        X_sample = X[:, idx]
        
        # Compute QR Decomposition to extract an orthonormal basis for the candidate subspace
        Q, R = sp_linalg.qr(X_sample, mode='economic', check_finite=False)

        # Check for Linear Independence using the last pivot element of R
        if R.shape[0] < d or np.abs(R[d-1, d-1]) < 1e-8:
            continue
            
        valid_iters += 1
        V_candidate = Q[:, :d]
        
        # Calculate orthogonal distances of all points to this candidate
        proj_sq_norms = np.sum((V_candidate.T @ X)**2, axis=0)
        dists = np.sqrt(np.maximum(X_sq_norms - proj_sq_norms, 0))
        
        # Count inliers
        num_inliers = np.sum(dists < threshold)
        
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_V = V_candidate

    if best_V is None:
        raise ValueError("Failed to find any non-degenerate samples. Check if data is severely rank-deficient.")
    
    # Refine the final subspace using standard PCA on the winning consensus set
    proj_sq_norms = np.sum((best_V.T @ X)**2, axis=0)
    dists = np.sqrt(np.maximum(X_sq_norms - proj_sq_norms, 0))
    inlier_indices = np.where(dists < threshold)[0]
    
    if len(inlier_indices) > d:
        try:
            U, S_refined, _ = svds(X[:, inlier_indices], k=d)
            # Resort the singular values and corresponding vectors in descending order
            sort_idx = np.argsort(S_refined)[::-1]
            
            if S_refined[sort_idx[-1]] > 1e-8: 
                best_V = U[:, sort_idx]
        except sp_linalg.LinAlgError:
            pass  # Fallback to the unrefined best_V if numerical edge-case fails
    
    run_time = perf_counter() - t_start
    print(f"RANSAC finished with {max_inliers} inliers, {attempts} attempts, {valid_iters} valid iterations, time taken: {run_time:.4f} seconds")

    return best_V, max_inliers, run_time


def RF(X, d, num_iters=1000, tol=1e-8):
    """
    Randomized-Find Algorithm for Robust Subspace Recovery
    Based on Hardt & Moitra, COLT 2013: "Algorithms and Hardness for Robust Subspace Recovery"
    
    Parameters:
    - X: (n, m) numpy array. Data matrix where n is ambient dimension, m is number of points.
    - d: int. Intrinsic dimension of the target subspace.
    - num_iters: int. Number of random sampling iterations to attempt.
    - tol: float. Tolerance for zero-checking singular values and null-space coefficients.
    
    Returns:
    - U_T: (n, d) numpy array. Orthogonal basis of the recovered subspace.
    - all_inliers: 1D numpy array of indices corresponding to the true inliers.
    - run_time: float. Time taken for the algorithm to complete.
    """
    t_start = perf_counter()
    n, m = X.shape
    if m < n:
        raise ValueError("Dataset must contain at least n points to sample an n x n matrix.")
    
    # Initialize a random generator for efficiency
    rng = np.random.default_rng()
    
    # Pre-compute X squared norms for the fast dataset-wide distance check
    X_sq_norms = np.linalg.norm(X, axis=0)**2

    known_inliers = set()
    
    for _ in range(num_iters):
        # Draw a random sample of exactly n points
        sample_idx = rng.choice(m, n, replace=False)
        X_sample = X[:, sample_idx]

        # Direct LAPACK SVD execution bypassing array copying overhead
        try:
            _, S_samp, Vh_samp = sp_linalg.svd(X_sample, overwrite_a=True, check_finite=False)
        except sp_linalg.LinAlgError:
            continue  # Handle rare numerical convergence failures safely
        
        # Check for linear dependence
        # An n x n matrix is linearly dependent if its smallest singular value is 0
        if S_samp[-1] < tol:  # The sample is linearly dependent
            # Extract a vector from the null space (the last row of Vh)
            null_vector = Vh_samp[-1, :]
            
            # Outliers are in general position, so they cannot participate
            # in the linear dependence. Non-zero coefficients strictly identify inliers.
            inlier_mask = np.abs(null_vector) > tol
            discovered_inliers = sample_idx[inlier_mask]
            
            # Add the newly discovered inliers to our running set
            known_inliers.update(discovered_inliers.tolist())
            
            # If we have accumulated at least d inliers, we can reconstruct the subspace
            if len(known_inliers) >= d:
                X_inliers = X[:, list(known_inliers)]
                U_in, S_in, _ = sp_linalg.svd(X_inliers, full_matrices=False)
                
                # Verify that the accumulated inliers actually span the full d-dimensional space
                if len(S_in) >= d and S_in[d-1] > tol:
                    U_T = U_in[:, :d]
                    
                    # Scan the entire dataset to find ALL points lying on this subspace
                    proj_sq_norms = np.sum((U_T.T @ X)**2, axis=0)
                    
                    # Pythagorean distance: ||x - UU^Tx||_2 = sqrt(||x||_2^2 - ||U^Tx||_2^2)
                    dists = np.sqrt(np.maximum(X_sq_norms - proj_sq_norms, 0))
                    
                    all_inliers = np.where(dists < tol)[0]

                    run_time = perf_counter() - t_start
                    print(f"Hardt-Moitra RF finished with {len(all_inliers)} inliers, time taken: {run_time:.4f} seconds")

                    return U_T, all_inliers, run_time
                    
    raise RuntimeError("Failed to find a linearly dependent sample. The fraction of inliers must be > d/n.")