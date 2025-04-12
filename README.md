Robust Subspace Recovery (RSR) under Adversarial and Noisy Corruptions

"intro_all_methods.py": The performance of various RSR methods across different corruption levels $\epsilon$. The considered methods are Tyler's M-estimator (TME) [1], Fast Median Subspace (FMS) [2], Geodesic Gradient Descent (GGD) [3], Randomized-Find (RF) [4], and the classic RANSAC algorithm [5]. In the toy example, the clean samples are drawn from $N(0, \Sigma^{\star})$ with $\mathrm{rank}(\Sigma^{\star}) = 10$, while outliers are drawn from $N(0, \widehat{\Sigma})$ with $\mathrm{rank}(\widehat{\Sigma}) = 2$. The Gaussian noise covariance is set to zero in these experiments. The subspace spanned by the outliers are chosen to be orthogonal to $\mathcal{S}^{\star}$. All nonzero eigenvalues of $\Sigma^{\star}$ are set to $1$, and the nonzero eigenvalues of $\widehat{\Sigma}$ are set to $10$. The ambient dimension is set to $d = 100$, and the total sample size is set to $n = 500$.

In the following, we provide some additional implementation details for this toy example.

Tyler’s M-estimator (TME) refers to formula (1.2) in Section 1 of [1]. In our implementation, the initialization is set as $\Sigma^{(0)} = I_d / d$. During iteration, any eigenvalue of $\Sigma^{(t)}$ smaller than $10^{-10}$ is set to zero. The final output consists of the eigenvectors corresponding to the nonzero eigenvalues of $\Sigma^{(T)}$, where $T$ is the total number of iterations. The algorithm terminates when $\Vert \Sigma^{(t)} - \Sigma^{(t-1)} \Vert_{2} < 10^{-4}$ or the total number of iterations $T$ reaches $10^2$.

Fast Median Subspace (FMS) corresponds to Algorithm 1 in Section 2 of [2]. All parameters are set to their default values, and the true subspace dimension $r^{\star}$ is provided to the algorithm. The algorithm terminates when the $\ell_2$-norm of the difference between two adjacent iterates is smaller than $10^{-4}$ or the total number of iterations $T$ reaches $10^2$.

Geodesic Gradient Descent (GGD) is implemented based on Algorithm 1 in Section 4 of [3]. The true subspace dimension $r^{\star}$ is supplied, with an initial step size $s_0 = 1 / d$, a tolerance $\tau = 10^{-4}$, a constant step interval length $K = 20$, and a shrink factor of $\frac{1}{2}$. The algorithm terminates when the angle between two consecutive subspaces is smaller than $10^{-4}$ or the total number of iterations $T$ reaches $10^3$.

Randomized-Find (RF) refers to Algorithm 1 in Section 2 of [4]. This method does not require any parameter tuning, and the maximum number of iterations is fixed at $10^3$.

The classic RANSAC is based on Algorithm 1 in Section 4 of [5]. In our setup, the true subspace dimension $r^{\star}$ is provided. The angular threshold is set to $\tau = 10^{-1}$, and the maximum number of iterations is fixed at $10^6$.

Our method, RANSAC+, consists of two stages. In the first stage, the threshold is set to $\eta_{\mathrm{thresh}} = 10^{-2}$, and the batch size $B$ increases by one at each iteration. In the second stage, we set the failure probability $\delta = 10^{-2}$, use $C' = \frac{2}{1 - \epsilon}$, and cap the number of iterations at $10^6$.



[1] T. Zhang, Robust subspace recovery by tyler’s m-estimator, Information and Inference: A Journal of the IMA, 5 (2016), pp. 1–21.

[2] G. Lerman and T. Maunu, Fast, robust and non-convex subspace recovery, Information and Inference: A Journal of the IMA, 7 (2018), pp. 277–336.

[3] T. Maunu, T. Zhang, and G. Lerman, A well-tempered landscape for non-convex robust subspace recovery, Journal of Machine Learning Research, 20 (2019), pp. 1–59.

[4] M. Hardt and A. Moitra, Algorithms and hardness for robust subspace recovery, in Conference on Learning Theory, PMLR, 2013, pp. 354–375.

[5] T. Maunu and G. Lerman, Robust subspace recovery with adversarial outliers, arXiv preprint arXiv:1904.03275, (2019).