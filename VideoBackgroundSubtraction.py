import os
import glob
import cv2
import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt
from RSR_methods import *


# ==========================================
# The Video Subspace Recovery Framework
# ==========================================
class VideoSubspaceRecoveryFramework:
    def __init__(self, input_path, is_folder=False, target_width=160, target_height=120, max_frames=200):
        """
        Initializes the framework by loading and preprocessing frames.
        Set `is_folder=True` for CDnet (image sequences).
        Set `is_folder=False` for standard video files (.mp4, .avi).
        """
        self.input_path = input_path
        self.w = target_width
        self.h = target_height
        self.d = self.w * self.h
        self.max_frames = max_frames
        
        if is_folder:
            self.X, self.original_shape = self._load_video_from_frames()
        else:
            self.X, self.original_shape = self._load_video_from_mp4()
            
        self.d, self.n = self.X.shape
        self.sigma_array = None
        print(f"Data Matrix X constructed: d={self.w}x{self.h}={self.d} (pixels), n={self.n} (frames).")

    def _load_video_from_mp4(self):
        """Loads video from a single file and flattens into columns."""
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video file: {self.input_path}")
            
        frames = []
        frame_idx = 0
        
        while cap.isOpened() and frame_idx < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (self.w, self.h), interpolation=cv2.INTER_AREA)
            flattened = resized.astype(np.float64) / 255.0
            
            frames.append(flattened.flatten())
            frame_idx += 1
            
        cap.release()
        return np.column_stack(frames), (self.h, self.w)

    def _load_video_from_frames(self):
        """Loads a sequence of images from a directory (for academic datasets)."""
        # Supports standard picture formats: jpg, png, bmp
        extensions = ('*.jpg', '*.png', '*.bmp')
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(self.input_path, ext)))
            
        image_paths = sorted(image_paths)[:self.max_frames]
        
        if not image_paths:
            raise FileNotFoundError(f"No valid images found in directory: {self.input_path}")
            
        frames = []
        for path in image_paths:
            frame = cv2.imread(path)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (self.w, self.h), interpolation=cv2.INTER_AREA)
            flattened = resized.astype(np.float64) / 255.0
            frames.append(flattened.flatten())
            
        return np.column_stack(frames), (self.h, self.w)

    def estimate_pixelwise_noise(self):
        """
        Estimates the Gaussian noise standard deviation for each pixel 
        using the robust Temporal Median Absolute Deviation (MAD).
        """
        print("Estimating pixelwise Gaussian noise profile...")
        # Initialize a high-performance random generator
        rng = np.random.default_rng()
        
        # Sample up to 200 random frames uniformly across the video
        sample_size = min(self.n, 200)
        sample_cols = rng.choice(self.n, size=sample_size, replace=False)
        
        # Random sampling
        X_sample = self.X[:, sample_cols] 
        median_frame = np.median(X_sample, axis=1, keepdims=True)
        abs_deviations = np.abs(X_sample - median_frame)
        mad = np.median(abs_deviations, axis=1)
        self.sigma_array = (1.4826) * mad  # $1.4826$ is the scaling factor for Gaussian consistency
        return self.sigma_array
    
    def run_experiment(self, method_func=None, r=None, **kwargs):
        """
        Runs the specified robust subspace recovery method.
        
        Parameters:
        - method_func: A callable function imported from RSR_methods file. 
                       If None, runs default `RANSAC+`.
        - r: Target rank (Required if using competitor algorithms like STE or FMS).
        - kwargs: Any algorithm-specific hyperparameters (e.g., th, eps, max_iter).
        """
        alg_name = method_func.__name__ if method_func else "RANSAC+"
        print(f"\nStarting robust subspace recovery using: [{alg_name}]...")
        
        t_start = perf_counter()

        # ==========================================
        # PATH A: RANSAC+
        # ==========================================
        if method_func is None:
            # Extract specific kwargs or fallback to defaults
            th = kwargs.get('th', None)
            eps = kwargs.get('eps', 0.2)
            
            if th is None:
                if self.sigma_array is None:
                    self.estimate_pixelwise_noise()
                th = np.sqrt(np.sum(self.sigma_array**2))
                print(f"--> Auto-calculated stopping threshold (via 200-frame sample): {th:.4f}")
            else:
                print(f"--> Using manual stopping threshold: {th}")
                
            # Run RANSAC+
            res_vecs, r, elapsed = RANSAC_PLUS(self.X, th=th, eps=eps)

        # ==========================================
        # PATH B: Competitor Algorithms
        # ==========================================
        else:
            if r is None:
                raise ValueError("You must provide a target rank 'r' for competitor algorithms.")
                
            # Execute the competitor method
            result = method_func(self.X, r, **kwargs)
            elapsed = perf_counter() - t_start
            
            # Competitors return different tuples. We dynamically extract the (D, d) basis matrix.
            if isinstance(result, tuple):
                for item in result:
                    if isinstance(item, np.ndarray) and item.shape == (self.d, r):
                        res_vecs = item
                        break
            else:
                res_vecs = result
                
        print(f"Algorithm finished in {elapsed:.2f} seconds.")
        print(f"--> Subspace Rank used/found: {r}")
        
        # ==========================================
        # Video Separation & Reconstruction
        # ==========================================
        # Reconstruct Background and Foreground using the recovered subspace
        # Zero-Memory Lazy Evaluators for Background and Foreground
        self.Background = LazyVideoMatrix(self.X, res_vecs, mode='background')
        self.Foreground = LazyVideoMatrix(self.X, res_vecs, mode='foreground')

        # # Reconstruct Background: project X onto the recovered subspace
        # self.Background = res_vecs @ (res_vecs.T @ self.X)
        # # Foreground is the residual (the adversarial corruptions)
        # self.Foreground = self.X - self.Background
        # # Boundary constraints for visualization safety
        # self.Background = np.clip(self.Background, 0, 1)
        # self.Foreground = np.abs(self.Foreground)
        print("Video separation complete.")

    def visualize_frame(self, frame_number):
        """Plots the Original, Background, and Foreground for a specific frame."""
        if frame_number >= self.n:
            raise ValueError(f"Frame number {frame_number} exceeds total frames {self.n}")
            
        orig = self.X[:, frame_number].reshape(self.original_shape)
        bg = self.Background[:, frame_number].reshape(self.original_shape)
        fg = self.Foreground[:, frame_number].reshape(self.original_shape)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(orig, cmap='gray'); axes[0].set_title(f"Original Frame #{frame_number}"); axes[0].axis('off')
        axes[1].imshow(bg, cmap='gray'); axes[1].set_title("Recovered Background"); axes[1].axis('off')
        axes[2].imshow(fg, cmap='hot'); axes[2].set_title("Recovered Foreground"); axes[2].axis('off')  # (Outliers)
        
        plt.tight_layout()
        plt.show()

    def save_output_video(self, output_dir="output_results"):
        """Saves the background and foreground streams back to disk as standard videos."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        bg_writer = cv2.VideoWriter(os.path.join(output_dir, 'background.mp4'), fourcc, 20.0, (self.w, self.h), False)
        fg_writer = cv2.VideoWriter(os.path.join(output_dir, 'foreground.mp4'), fourcc, 20.0, (self.w, self.h), False)
        
        print("\nSaving output videos...")
        for i in range(self.n):
            bg_frame = (self.Background[:, i].reshape(self.original_shape) * 255).astype(np.uint8)
            fg_frame = (np.clip(self.Foreground[:, i].reshape(self.original_shape), 0, 1) * 255).astype(np.uint8)
            
            bg_writer.write(bg_frame)
            fg_writer.write(fg_frame)
            
        bg_writer.release()
        fg_writer.release()
        print(f"Resulting videos successfully saved to: {output_dir}/")


# ==========================================
# Memory-Decoupled Lazy Evaluator for Video Reconstruction
# ==========================================
class LazyVideoMatrix:
    def __init__(self, X, res_vecs, mode='foreground'):
        """
        Memory-decoupled lazy evaluator for low-rank video reconstructions.
        Bypasses gigabyte-scale matrix allocations.
        """
        self.X = X
        self.res_vecs = res_vecs  # Shape: (D, r)
        self.mode = mode
        
        # Precompute the tiny projection coefficients matrix (r x N)
        self.H = res_vecs.T @ X 
        
        # Expose the shape so framework property checks still work
        self.shape = X.shape

    def __getitem__(self, key):
        """Intercepts the [:, frame_idx] call to compute values on the fly."""
        if isinstance(key, tuple) and len(key) == 2:
            row_slice, col_idx = key
            if row_slice == slice(None):
                # Compute only the single 1D column needed for this specific frame
                bg_col = self.res_vecs @ self.H[:, col_idx]
                
                if self.mode == 'background':
                    return np.clip(bg_col, 0, 1)
                else:  # foreground
                    return np.abs(self.X[:, col_idx] - bg_col)
                    
        raise NotImplementedError("Only column slicing [:, frame_idx] is supported for lazy memory management.")


# ==========================================
# CDnet Evaluator
# ==========================================
class CDnetEvaluator:
    @staticmethod
    def read_temporal_roi(roi_txt_path):
        """
        Reads the temporalROI.txt file to get the valid start and end frames.
        """
        if not os.path.exists(roi_txt_path):
            raise FileNotFoundError(f"Could not find {roi_txt_path}")
            
        with open(roi_txt_path, 'r') as f:
            content = f.read().strip().split()
            start_frame = int(content[0])
            end_frame = int(content[1])
            
        return start_frame, end_frame

    @staticmethod
    def evaluate_sequence(framework, gt_dir, roi_txt_path, dynamic_threshold=None):
        """
        Evaluates the entire video sequence according to official CDnet rules.
        Accumulates TP, FP, and FN across the Temporal ROI and Spatial ROI.
        
        Parameters:
        - framework: Your instantiated VideoSubspaceRecoveryFramework (after running experiment).
        - gt_dir: Path to the CDnet 'groundtruth' folder containing the .png files.
        - roi_txt_path: Path to the 'temporalROI.txt' file.
        """
        start_frame, end_frame = CDnetEvaluator.read_temporal_roi(roi_txt_path)
        print(f"Evaluating Sequence... Temporal ROI: Frames {start_frame} to {end_frame}")
        
        # Initialize global sequence counters
        total_TP = 0
        total_FP = 0
        total_FN = 0

        # Create a morphological kernel for post-processing
        kernel = np.ones((5, 5), np.uint8)  # 3
        
        # CDnet filenames are 1-indexed (e.g., gt000300.png)
        for frame_idx in range(start_frame, end_frame + 1):
            # Convert to 0-based index to access your framework's numpy array
            arr_idx = frame_idx - 1 
            
            # Safeguard in case the framework processed fewer frames than the CDnet max
            if arr_idx >= framework.Foreground.shape[1]:
                break
                
            # Load the corresponding Ground Truth image
            gt_filename = f"gt{frame_idx:06d}.png"
            gt_path = os.path.join(gt_dir, gt_filename)
            
            if not os.path.exists(gt_path):
                print(f"Warning: Ground truth missing for frame {frame_idx}")
                continue
                
            # Load GT and resize to match your framework's resolution (using nearest neighbor)
            gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            gt_resized = cv2.resize(gt_img, (framework.w, framework.h), interpolation=cv2.INTER_NEAREST)
            true_mask_raw = gt_resized.flatten()
            
            # Get your algorithm's estimated foreground for this frame
            est_foreground = framework.Foreground[:, arr_idx]
            est_fg_abs = np.abs(est_foreground)
            
            # Apply dynamic noise-based thresholding (if not manually overridden)
            if dynamic_threshold is None:
                mad = np.median(np.abs(est_fg_abs - np.median(est_fg_abs)))
                threshold = 5 * 1.4826 * mad  # 3
            else:
                threshold = dynamic_threshold
            
            # Binarize the estimated foreground using the threshold
            est_binary_1d = (est_fg_abs > threshold).astype(np.uint8)

            # Reshape back to 2D for morphological operations
            est_binary_2d = est_binary_1d.reshape((framework.h, framework.w))
            
            # Apply morphological opening to clean up small noise blobs (common in CDnet evaluations)
            clean_binary_2d = cv2.morphologyEx(est_binary_2d, cv2.MORPH_OPEN, kernel)

            # Flatten back to 1D to compare with the flattened ground truth
            clean_binary_1d = clean_binary_2d.flatten()

            # --- Enforce CDnet Spatial ROI Rules ---
            is_foreground = (true_mask_raw == 255)
            is_background = (true_mask_raw == 0)
            # Note: Pixels labeled 85 (Outside ROI), 50 (Shadow), and 170 (Unknown) 
            # are mathematically ignored because they are neither 255 nor 0.
            
            # Accumulate metrics
            total_TP += np.sum(clean_binary_1d & is_foreground)
            total_FN += np.sum(~clean_binary_1d & is_foreground)
            total_FP += np.sum(clean_binary_1d & is_background)

        # Calculate final sequence-level metrics
        precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0.0
        recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0.0
        f_measure = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f_measure
    
    @staticmethod
    def plot_error_map(framework, frame_number, gt_path, multiplier=5, kernel_size=5):
        """
        Visually isolates and displays False Positives (Red), True Positives (Green), 
        and False Negatives (Blue) for a specific frame

        Parameters:
        - framework: The instantiated VideoSubspaceRecoveryFramework (after running experiment).
        - frame_number: The specific frame index to analyze.
        - gt_path: Path to the CDnet 'groundtruth' file for the specific frame.
        - multiplier: The multiplier for the MAD-based threshold (default=5).
        - kernel_size: The size of the morphological kernel for post-processing (default=5).
        """
        # Get the algorithm's output
        est_fg_1d = np.abs(framework.Foreground[:, frame_number])
        
        # Apply MAD-based thresholding to get a binary mask of the estimated foreground
        mad = np.median(np.abs(est_fg_1d - np.median(est_fg_1d)))
        threshold = multiplier * 1.4826 * mad
        
        est_binary_1d = (est_fg_1d > threshold).astype(np.uint8)
        est_binary_2d = est_binary_1d.reshape((framework.h, framework.w))
        
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        clean_binary_2d = cv2.morphologyEx(est_binary_2d, cv2.MORPH_OPEN, kernel)
        
        # Load the corresponding Ground Truth
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        gt_resized = cv2.resize(gt_img, (framework.w, framework.h), interpolation=cv2.INTER_NEAREST)
        
        is_foreground = (gt_resized == 255)
        is_background = (gt_resized == 0)
        binary_gt = (is_foreground * 255).astype(np.uint8)  # convert to binary for visualization
        
        # Calculate exactly where the algorithm messed up
        # True Positives: We guessed 1, GT says 255
        TP_mask = (clean_binary_2d == 1) & is_foreground
        
        # False Positives: We guessed 1, GT says 0 (THE Precision Killer)
        FP_mask = (clean_binary_2d == 1) & is_background
        
        # False Negatives: We guessed 0, GT says 255 (The Recall Killer)
        FN_mask = (clean_binary_2d == 0) & is_foreground

        # Build a error map for visualization
        # Original image in grayscale
        orig_gray = framework.X[:, frame_number].reshape((framework.h, framework.w))
        debug_img = cv2.cvtColor((orig_gray * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        
        # Paint True Positives Green (What we got right)
        debug_img[TP_mask] = [0, 255, 0]
        
        # Paint False Positives Red (What is killing the Precision)
        debug_img[FP_mask] = [255, 0, 0]
        
        # Paint False Negatives Blue (What is killing the Recall)
        debug_img[FN_mask] = [0, 0, 255]

        # Plotting
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(binary_gt, cmap='gray')
        axes[0].set_title(f"Ground Truth #{frame_number}", fontsize=12)
        axes[0].axis('off')
        
        axes[1].imshow(clean_binary_2d, cmap='gray')
        axes[1].set_title("Generated Binary Mask", fontsize=12)
        axes[1].axis('off')
        
        axes[2].imshow(debug_img)
        axes[2].set_title("Error Map", fontsize=12)  # (Red = False Positives, Green = True Positives, Blue = False Negatives)
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()


# ==========================================
if __name__ == "__main__":
    # # --- Example 1: Using a standard .mp4 video ---
    # video_file = "traffic.mp4" 
    
    # # Create a synthetic test video if 'traffic.mp4' isn't found locally
    # if not os.path.exists(video_file):
    #     print(f"Creating a synthetic video '{video_file}' for framework verification...")
    #     dummy_capsule = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (320, 240))
    #     for f in range(100):
    #         img = np.ones((240, 320, 3), dtype=np.uint8) * 120
    #         cv2.rectangle(img, (20 + f*2, 100), (50 + f*2, 130), (255, 255, 255), -1) 
    #         noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
    #         img = cv2.add(img, noise)
    #         dummy_capsule.write(img)
    #     dummy_capsule.release()

    # # Initialize the framework for a video file (is_folder=False)
    # framework = VideoSubspaceRecoveryFramework(
    #     input_path=video_file, 
    #     is_folder=False, 
    #     target_width=160, 
    #     target_height=120, 
    #     max_frames=100
    # )
    
    # # Execute recovery
    # framework.run_experiment(th=None, eps=0.2)
    
    # # Visualize a middle frame
    # framework.visualize_frame(frame_number=50)
    
    # # Save the separated streams as .mp4 files
    # framework.save_output_video()

    # --- Example 2: Using CDnet dataset ---
    dataset_folder = "./CDnet/badWeather/snowFall"   # cameraJitter/sidewalk or badWeather/snowFall
    input_folder = os.path.join(dataset_folder, "input")
    roi_txt_path=os.path.join(dataset_folder, "temporalROI.txt")
    _, end_frame = CDnetEvaluator.read_temporal_roi(roi_txt_path)
    if os.path.exists(input_folder):
        cdnet_framework = VideoSubspaceRecoveryFramework(
            input_path=input_folder, 
            is_folder=True, 
            target_width=720,
            target_height=480,
            max_frames=end_frame
        )
        cdnet_framework.run_experiment(method_func=None, eps=0.4)  # method_func=None, STE, FMS, GGD, RPCA
        cdnet_framework.visualize_frame(frame_number=900)
        CDnetEvaluator.plot_error_map(cdnet_framework, frame_number=900, gt_path=os.path.join(dataset_folder, "groundtruth", "gt000900.png"), multiplier=5, kernel_size=5)
        cdnet_framework.visualize_frame(frame_number=1200)
        CDnetEvaluator.plot_error_map(cdnet_framework, frame_number=1200, gt_path=os.path.join(dataset_folder, "groundtruth", "gt001200.png"), multiplier=5, kernel_size=5)
        cdnet_framework.visualize_frame(frame_number=1500)
        CDnetEvaluator.plot_error_map(cdnet_framework, frame_number=1500, gt_path=os.path.join(dataset_folder, "groundtruth", "gt001500.png"), multiplier=5, kernel_size=5)
        cdnet_framework.visualize_frame(frame_number=1800)
        CDnetEvaluator.plot_error_map(cdnet_framework, frame_number=1800, gt_path=os.path.join(dataset_folder, "groundtruth", "gt001800.png"), multiplier=5, kernel_size=5)
        cdnet_framework.visualize_frame(frame_number=2100)
        CDnetEvaluator.plot_error_map(cdnet_framework, frame_number=2100, gt_path=os.path.join(dataset_folder, "groundtruth", "gt002100.png"), multiplier=5, kernel_size=5)
        cdnet_framework.visualize_frame(frame_number=2400)
        CDnetEvaluator.plot_error_map(cdnet_framework, frame_number=2400, gt_path=os.path.join(dataset_folder, "groundtruth", "gt002400.png"), multiplier=5, kernel_size=5)
        cdnet_framework.visualize_frame(frame_number=2700)
        CDnetEvaluator.plot_error_map(cdnet_framework, frame_number=2700, gt_path=os.path.join(dataset_folder, "groundtruth", "gt002700.png"), multiplier=5, kernel_size=5)
        cdnet_framework.visualize_frame(frame_number=3000)
        CDnetEvaluator.plot_error_map(cdnet_framework, frame_number=3000, gt_path=os.path.join(dataset_folder, "groundtruth", "gt003000.png"), multiplier=5, kernel_size=5)
        cdnet_framework.visualize_frame(frame_number=3300)
        CDnetEvaluator.plot_error_map(cdnet_framework, frame_number=3300, gt_path=os.path.join(dataset_folder, "groundtruth", "gt003300.png"), multiplier=5, kernel_size=5)
        p, r, f1 = CDnetEvaluator.evaluate_sequence(
            framework=cdnet_framework,
            gt_dir=os.path.join(dataset_folder, "groundtruth"),
            roi_txt_path=roi_txt_path,
            )
        print(f"Final Video Separation Performance -> F-Measure: {f1:.4f} (Precision: {p:.4f}, Recall: {r:.4f})")
