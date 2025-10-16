"""
Enhanced Boundary & Curvature Evaluation Utilities for 3D Point Cloud Segmentation
- Scalable band-based boundary detection using an external k-NN function (knn_fn)
- Per-sample curvature normalization (percentiles) and curvature binning
- Vectorized confusion accumulation (fast, safe for large point clouds)
- Multi-width boundary reporting, ignore-index handling, and stability guards
"""

from typing import Callable, Dict, List, Optional
import numpy as np
import torch

def _accumulate_confusion_numpy(preds_flat: np.ndarray, labels_flat: np.ndarray, num_classes: int, conf: np.ndarray):
    """Vectorized accumulation into confusion matrix using bincount."""
    if preds_flat.size == 0:
        return conf
    # Ensure labels and predictions are within the valid range [0, num_classes-1] before calculating index
    mask = (preds_flat >= 0) & (preds_flat < num_classes) & (labels_flat >= 0) & (labels_flat < num_classes)
    if not np.any(mask):
        return conf
    
    # Calculate index only for valid entries
    idx = preds_flat[mask] * num_classes + labels_flat[mask]
    
    counts = np.bincount(idx, minlength=num_classes * num_classes)
    conf += counts.reshape((num_classes, num_classes))
    return conf

class BoundaryAwareMetrics:
    def __init__(self, num_classes: int, boundary_width: float = 0.1, ignore_index: int = 255, k_boundary: int = 32):
        self.num_classes = int(num_classes)
        self.boundary_width = float(boundary_width)
        self.ignore_index = int(ignore_index)
        self.k_boundary = int(k_boundary)
        self.reset()

    def reset(self):
        """Resets all metric accumulators to zero for a new evaluation epoch."""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self._accum_preds: List[np.ndarray] = []
        self._accum_labels: List[np.ndarray] = []
        self._accum_xyz: List[np.ndarray] = []
        self._accum_full_labels: List[np.ndarray] = []

    def _ensure_knn_fn(self, knn_fn: Optional[Callable]):
        if knn_fn is None:
            raise ValueError("A `knn_fn` (e.g., your model's knn_blocked) must be provided for scalable evaluation.")

    def _compute_boundary_mask_band(self, xyz: torch.Tensor, labels: torch.Tensor, knn_fn: Callable, k: int, boundary_width: float) -> torch.Tensor:
        """Computes a 'trimap' style boundary mask based on geometric distance."""
        B, N, _ = xyz.shape
        device = xyz.device

        nn_idx = knn_fn(xyz, xyz, k)
        batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(-1, N, k)
        neigh_xyz = xyz[batch_idx, nn_idx]
        neigh_labels = labels[batch_idx, nn_idx]

        center_labels = labels.unsqueeze(-1)
        # A point has a different neighbor if the neighbor is not the same class AND not an ignored class (original 0)
        diff_mask = (neigh_labels != center_labels) & (center_labels != 0) & (neigh_labels != 0)

        if diff_mask.any():
            dists = torch.norm(neigh_xyz - xyz.unsqueeze(2), dim=-1)
            dists_masked = torch.where(diff_mask, dists, torch.full_like(dists, 1e6))
            min_dist, _ = dists_masked.min(dim=-1)
            return min_dist <= boundary_width
        else:
            return torch.zeros(B, N, dtype=torch.bool, device=device)

    def update(self, predictions: torch.Tensor, labels: torch.Tensor, xyz: torch.Tensor):
        """Accumulates results from a batch for later computation."""
        if predictions.dim() == 3:
            # Input is logits [B, C, N] or [B, N, C], convert to labels [B, N]
            if predictions.shape[1] == self.num_classes:
                preds = predictions.argmax(dim=1)
            else:
                preds = predictions.argmax(dim=-1)
        else:
            preds = predictions
        
        preds_np = preds.detach().cpu().numpy().astype(np.int64)
        labels_np = labels.detach().cpu().numpy().astype(np.int64)
        xyz_np = xyz.detach().cpu().numpy().astype(np.float32)

        # Original ignore label is 0
        valid_mask = (labels_np != 0) 
        # Remap ground truth labels [1-8] to [0-7] for confusion matrix
        remapped_labels_np = labels_np - 1

        for b in range(preds.shape[0]):
            mask_b = valid_mask[b]
            # Accumulate remapped preds and labels for valid points
            self._accum_preds.append(preds_np[b][mask_b])
            self._accum_labels.append(remapped_labels_np[b][mask_b])
            # Accumulate full original data for boundary calculation
            self._accum_xyz.append(xyz_np[b])
            self._accum_full_labels.append(labels_np[b]) 
            
            pflat = preds_np[b][mask_b]
            lflat = remapped_labels_np[b][mask_b]
            self.confusion_matrix = _accumulate_confusion_numpy(pflat, lflat, self.num_classes, self.confusion_matrix)

    def compute_metrics(self, knn_fn: Callable, boundary_widths: Optional[List[float]] = None, k_boundary: Optional[int] = None) -> Dict[str, float]:
        self._ensure_knn_fn(knn_fn)
        boundary_widths = boundary_widths or [self.boundary_width]
        k_boundary = k_boundary or self.k_boundary
        
        metrics: Dict[str, float] = {}
        
        iou_per_class = self._calculate_iou(self.confusion_matrix)
        metrics['mIoU'] = float(np.nanmean(iou_per_class))
        
        for bw in boundary_widths:
            b_conf = np.zeros_like(self.confusion_matrix)
            for i in range(len(self._accum_xyz)):
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                xyz_tensor = torch.from_numpy(self._accum_xyz[i]).unsqueeze(0).to(device)
                # Use original labels (with 0 as ignore) for boundary detection
                labels_tensor = torch.from_numpy(self._accum_full_labels[i]).unsqueeze(0).to(device)
                
                b_mask = self._compute_boundary_mask_band(xyz_tensor, labels_tensor, knn_fn, k=k_boundary, boundary_width=bw).squeeze(0).cpu().numpy()
                
                # Reconstruct full prediction array to align with full boundary mask
                full_preds_sample = np.full_like(self._accum_full_labels[i], -1) # Use -1 for invalid
                full_labels_sample = self._accum_full_labels[i]
                
                valid_mask_sample = (full_labels_sample != 0)
                full_preds_sample[valid_mask_sample] = self._accum_preds[i]

                # Remap the full original labels for confusion matrix calculation
                full_labels_remapped = full_labels_sample - 1
                
                # Final mask: points that are on a boundary AND have a valid original label
                final_b_mask = b_mask & valid_mask_sample

                preds_b = full_preds_sample[final_b_mask]
                labels_b = full_labels_remapped[final_b_mask]
                
                b_conf = _accumulate_confusion_numpy(preds_b, labels_b, self.num_classes, b_conf)

            b_iou = self._calculate_iou(b_conf)
            metrics[f'boundary_mIoU_w{bw:.3f}'] = float(np.nanmean(b_iou))

        return metrics

    def _calculate_iou(self, cm: np.ndarray) -> np.ndarray:
        intersection = np.diag(cm)
        union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
        iou = np.divide(intersection, union, out=np.full_like(intersection, np.nan, dtype=float), where=union != 0)
        return iou

class CurvatureAwareEvaluator:
    def __init__(self, num_classes: int, curvature_bins: int = 5):
        self.num_classes = int(num_classes)
        self.curvature_bins = int(curvature_bins)
    
    def gather_neighbors(self, data: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """A robust helper for batch-aware gathering of neighbor data on the GPU."""
        B, _, D = data.shape
        _, N_query, K = indices.shape
        batch_idx = torch.arange(B, device=data.device).view(B, 1, 1).expand(-1, N_query, K)
        return data[batch_idx, indices]
    
    def evaluate(self, predictions: torch.Tensor, labels: torch.Tensor, xyz: torch.Tensor, knn_fn: Callable, k: int = 16, eps: float = 1e-8) -> Dict[str, float]:
        if predictions.dim() == 3:
            if predictions.shape[1] == self.num_classes:
                predictions = predictions.argmax(dim=1)
            else:
                predictions = predictions.argmax(dim=-1)

        B, N = predictions.shape[:2]
        
        with torch.no_grad():
            nn_idx = knn_fn(xyz, xyz, k)
            neigh_xyz = self.gather_neighbors(xyz, nn_idx)
            centered = neigh_xyz - xyz.unsqueeze(2)
            cov = torch.matmul(centered.transpose(-1, -2), centered) / float(k)
            eig = torch.linalg.eigvalsh(cov)
            curvature = eig[..., 0] / (eig.sum(dim=-1) + eps)
        
        curv_norm = torch.zeros_like(curvature)
        for b in range(B):
            c = curvature[b]
            lo, hi = torch.quantile(c, 0.01), torch.quantile(c, 0.99)
            if hi - lo > eps:
                curv_norm[b] = ((c - lo) / (hi - lo)).clamp(0.0, 1.0)
        
        results: Dict[str, float] = {}
        thresholds = torch.linspace(0.0, 1.0, self.curvature_bins + 1, device=curv_norm.device)

        for i in range(self.curvature_bins):
            low, high = thresholds[i], thresholds[i+1]
            mask = (curv_norm >= low) & (curv_norm <= high) if i == self.curvature_bins - 1 else (curv_norm >= low) & (curv_norm < high)
            
            valid_mask = (labels != 255) # Use the same ignore_index as the main loss
            final_mask = mask & valid_mask

            if final_mask.any():
                pred_bin = predictions[final_mask]
                label_bin = labels[final_mask]
                
                pred_np = pred_bin.cpu().numpy()
                label_np = label_bin.cpu().numpy()
                
                cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
                cm = _accumulate_confusion_numpy(pred_np, label_np, self.num_classes, cm)
                
                iou_bin = BoundaryAwareMetrics(self.num_classes)._calculate_iou(cm)
                results[f'mIoU_curvature_bin_{i}'] = float(np.nanmean(iou_bin))
                
        return results





# from typing import Callable, Dict, List, Optional, Tuple
#error code
# import time
# import numpy as np
# import torch
# import torch.nn.functional as F


# def _accumulate_confusion_numpy(preds_flat: np.ndarray, labels_flat: np.ndarray, num_classes: int, conf: np.ndarray):
#     """Vectorized accumulation into confusion matrix using bincount.
#     preds_flat, labels_flat are 1-D numpy arrays (already masked for ignore).
#     conf is shape (num_classes, num_classes)
#     """
#     if preds_flat.size == 0:
#         return conf
#     idx = preds_flat * num_classes + labels_flat
#     counts = np.bincount(idx, minlength=num_classes * num_classes)
#     conf += counts.reshape((num_classes, num_classes))
#     return conf


# class BoundaryAwareMetrics:
#     def __init__(self, num_classes: int, boundary_width: float = 0.1, ignore_index: int = -100, k_boundary: int = 32):
#         self.num_classes = int(num_classes)
#         self.boundary_width = float(boundary_width)
#         self.ignore_index = int(ignore_index)
#         self.k_boundary = int(k_boundary)
#         self.reset()

#     def reset(self):
#         self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
#         # We'll track boundary confusion per width on compute_metrics stage
#         self.total_points_count = 0
#         # store per-batch accumulations as lists to avoid premature flattening
#         self._accum_preds = []  # list of numpy arrays (preds_flat per batch)
#         self._accum_labels = []
#         self._accum_xyz = []

#     # ---------- Core helpers ----------
#     def _ensure_knn_fn(self, knn_fn: Optional[Callable]):
#         if knn_fn is None:
#             raise ValueError("knn_fn (query->support kNN) must be provided for scalable evaluation.\n"
#                              "Pass your knn_blocked or a ball-query function.")

#     def compute_boundary_mask_band(self, xyz: torch.Tensor, labels: torch.Tensor, knn_fn: Callable, k: int = None, boundary_width: float = None) -> torch.Tensor:
       
#         k = self.k_boundary if k is None else int(k)
#         boundary_width = self.boundary_width if boundary_width is None else float(boundary_width)
#         self._ensure_knn_fn(knn_fn)

#         B, N, _ = xyz.shape
#         device = xyz.device

#         nn_idx = knn_fn(xyz, xyz, k)  # (B, N, k)
#         batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(-1, N, k)

#         neigh_xyz = xyz[batch_idx, nn_idx]      # (B, N, k, 3)
#         neigh_labels = labels[batch_idx, nn_idx]  # (B, N, k)

#         center_labels = labels.unsqueeze(-1).expand_as(neigh_labels)
#         diff_mask = (neigh_labels != center_labels) & (neigh_labels != self.ignore_index)

#         if diff_mask.any():
#             dists = torch.norm(neigh_xyz - xyz.unsqueeze(2), dim=-1)  # (B, N, k)
#             large = 1e6
#             dists_masked = torch.where(diff_mask, dists, torch.full_like(dists, large))
#             min_dist, _ = dists_masked.min(dim=-1)
#             return (min_dist <= boundary_width)
#         else:
#             return torch.zeros(B, N, dtype=torch.bool, device=device)

#     def compute_curvature_weights(self, xyz: torch.Tensor, knn_fn: Callable, k: int = 16, eps: float = 1e-8) -> torch.Tensor:
#         """
#         Compute a normalized curvature weight per point in [0,1] using local covariance eigenvalues.
#         Normalization is performed per-sample using percentile clipping (robust to outliers).

#         Args:
#             xyz: (B,N,3)
#             knn_fn: function(query_xyz, support_xyz, k) -> (B,N,k)
#             k: neighbors
#         Returns:
#             curv_norm: (B,N) float tensor in [0,1]
#         """
#         self._ensure_knn_fn(knn_fn)
#         k = int(k)
#         B, N, _ = xyz.shape
#         device = xyz.device

#         nn_idx = knn_fn(xyz, xyz, k)
#         batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(-1, N, k)
#         neigh_xyz = xyz[batch_idx, nn_idx]
#         centered = neigh_xyz - xyz.unsqueeze(2)
#         cov = torch.matmul(centered.transpose(-1, -2), centered) / float(k)  # (B,N,3,3)

#         eig = torch.linalg.eigvalsh(cov)  # (B,N,3) ascending
#         curvature = eig[..., 0] / (eig.sum(dim=-1) + eps)  # (B,N)

#         cur_norm = torch.zeros_like(curvature)
#         # Percentile-based normalization per sample
#         for b in range(B):
#             c = curvature[b]
#             lo = float(torch.quantile(c, 0.01))
#             hi = float(torch.quantile(c, 0.99))
#             if hi - lo < 1e-6:
#                 cur_norm[b] = torch.zeros_like(c)
#             else:
#                 cur_norm[b] = ((c - lo) / (hi - lo + eps)).clamp(0.0, 1.0)
#         return cur_norm

#     # ---------- API: batch-wise accumulation ----------
#     def update(self, predictions: torch.Tensor, labels: torch.Tensor, xyz: torch.Tensor, knn_fn: Optional[Callable] = None):
#         """
#         Accumulate predictions/labels/coords from a batch. Predictions may be logits (B,N,C) or ints (B,N).
#         This stores flattened arrays for later global metric computation (multi-width boundary computation).
#         Must pass knn_fn for boundary/curvature operations during final evaluation.
#         """
#         # convert to prediction labels
#         if predictions.dim() == 3:
#             preds = predictions.argmax(dim=-1)
#         else:
#             preds = predictions

#         B, N = preds.shape[:2]

#         preds_np = preds.detach().cpu().numpy().astype(np.int64)
#         labels_np = labels.detach().cpu().numpy().astype(np.int64)
#         xyz_np = xyz.detach().cpu().numpy().astype(np.float32)

#         # mask out ignore labels when accumulating
#         valid_mask = (labels_np != self.ignore_index)

#         # Flatten batch-wise to 1D arrays but keep per-batch storage for boundary ops
#         for b in range(B):
#             mask_b = valid_mask[b]
#             pflat = preds_np[b][mask_b]
#             lflat = labels_np[b][mask_b]
#             self._accum_preds.append(pflat)
#             self._accum_labels.append(lflat)
#             self._accum_xyz.append(xyz_np[b])  # store full xyz per sample; labels mask used later
#             self.total_points_count += int(mask_b.sum())

#         # update global confusion fast (vectorized)
#         for b in range(B):
#             mask_b = (labels_np[b] != self.ignore_index)
#             pflat = preds_np[b][mask_b].reshape(-1)
#             lflat = labels_np[b][mask_b].reshape(-1)
#             self.confusion_matrix = _accumulate_confusion_numpy(pflat, lflat, self.num_classes, self.confusion_matrix)

#     # ---------- compute final metrics (after all updates) ----------
#     def compute_metrics(self, knn_fn: Callable, boundary_widths: Optional[List[float]] = None, k_boundary: Optional[int] = None) -> Dict[str, float]:
#         """
#         Compute standard + boundary metrics. boundary_widths: list of widths to evaluate (meters). Returns a dict with
#         standard mIoU/mAcc and boundary metrics for each width. Requires knn_fn.
#         """
#         self._ensure_knn_fn(knn_fn)
#         if boundary_widths is None:
#             boundary_widths = [self.boundary_width]
#         k_boundary = self.k_boundary if k_boundary is None else int(k_boundary)

#         metrics: Dict[str, float] = {}

#         # Standard metrics (mIoU, per-class IoU, overall accuracy, mAcc)
#         total = self.confusion_matrix.sum()
#         correct = np.diag(self.confusion_matrix).sum()
#         metrics['overall_accuracy'] = float(correct / total) if total > 0 else 0.0

#         iou_per_class = []
#         class_acc = []
#         for i in range(self.num_classes):
#             inter = self.confusion_matrix[i, i]
#             union = int(self.confusion_matrix[i, :].sum() + self.confusion_matrix[:, i].sum() - inter)
#             iou = float(inter / union) if union > 0 else 0.0
#             iou_per_class.append(iou)
#             gt_count = int(self.confusion_matrix[:, i].sum())
#             acc = float(inter / gt_count) if gt_count > 0 else 0.0
#             class_acc.append(acc)
#             metrics[f'IoU_class_{i}'] = iou
#             metrics[f'Accuracy_class_{i}'] = acc

#         metrics['mIoU'] = float(np.mean(iou_per_class)) if len(iou_per_class) > 0 else 0.0
#         metrics['mAcc'] = float(np.mean(class_acc)) if len(class_acc) > 0 else 0.0

#         # Boundary metrics per width
#         # We'll compute per-sample boundary masks using knn_fn and then aggregate boundary confusion
#         for bw in boundary_widths:
#             bw = float(bw)
#             # boundary confusion per class
#             bconf = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
#             b_total_points = 0

#             for idx_sample in range(len(self._accum_preds)):
#                 sample_preds = self._accum_preds[idx_sample]
#                 sample_labels = self._accum_labels[idx_sample]
#                 sample_xyz = torch.from_numpy(self._accum_xyz[idx_sample]).unsqueeze(0).float().to(next((torch.cuda._initialized and [torch.device('cuda')] or [torch.device('cpu')])[0]))
            
#                 p_t = torch.from_numpy(sample_preds).long().unsqueeze(0).to(sample_xyz.device)  # (1, M)
#                 l_t = torch.from_numpy(sample_labels).long().unsqueeze(0).to(sample_xyz.device)

#                 # Compute boundary mask on masked points using knn_fn
#                 mask_b = self.compute_boundary_mask_band(sample_xyz, l_t, knn_fn, k=k_boundary, boundary_width=bw).squeeze(0)  # (M,)

#                 # gather boundary preds/labels
#                 pred_b = p_t.squeeze(0)[mask_b].cpu().numpy().astype(np.int64)
#                 label_b = l_t.squeeze(0)[mask_b].cpu().numpy().astype(np.int64)

#                 b_total_points += int(mask_b.sum().item())
#                 # accumulate into bconf
#                 bconf = _accumulate_confusion_numpy(pred_b, label_b, self.num_classes, bconf)

#             # compute boundary metrics for this width
#             boundary_iou_per_class = []
#             boundary_f1_per_class = []
#             for i in range(self.num_classes):
#                 inter = bconf[i, i]
#                 union = int(bconf[i, :].sum() + bconf[:, i].sum() - inter)
#                 biou = float(inter / union) if union > 0 else 0.0
#                 boundary_iou_per_class.append(biou)

#                 tp = inter
#                 fp = int(bconf[i, :].sum()) - tp
#                 fn = int(bconf[:, i].sum()) - tp
#                 prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
#                 rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
#                 f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
#                 boundary_f1_per_class.append(f1)

#             metrics[f'boundary_mIoU_w{bw:.3f}'] = float(np.mean(boundary_iou_per_class)) if len(boundary_iou_per_class) > 0 else 0.0
#             metrics[f'boundary_mF1_w{bw:.3f}'] = float(np.mean(boundary_f1_per_class)) if len(boundary_f1_per_class) > 0 else 0.0
#             metrics[f'boundary_point_percentage_w{bw:.3f}'] = float(b_total_points / max(1, self.total_points_count))

#         return metrics


# class CurvatureAwareEvaluator:
#     def __init__(self, num_classes: int, curvature_bins: int = 5):
#         self.num_classes = int(num_classes)
#         self.curvature_bins = int(curvature_bins)

#     def evaluate(self, predictions: torch.Tensor, labels: torch.Tensor, xyz: torch.Tensor, knn_fn: Callable, k: int = 16) -> Dict[str, float]:
#         """
#         Compute per-curvature-bin accuracy and mIoU. Predictions may be logits (B,N,C) or labels (B,N).
#         Returns dict of metrics: accuracy_curvature_bin_{i}, mIoU_curvature_bin_{i}
#         """
#         if predictions.dim() == 3:
#             predictions = predictions.argmax(dim=-1)

#         B, N = predictions.shape[:2]
#         curv = BoundaryAwareMetrics(self.num_classes).compute_curvature_weights(xyz, knn_fn, k=k)
#         results: Dict[str, float] = {}

#         # bin thresholds per-sample are [0..1] since compute_curvature_weights normalizes per sample
#         thresholds = torch.linspace(0.0, 1.0, self.curvature_bins + 1, device=curv.device)

#         for i in range(self.curvature_bins):
#             low, high = thresholds[i].item(), thresholds[i + 1].item()
#             if i == self.curvature_bins - 1:
#                 mask = (curv >= low) & (curv <= high)
#             else:
#                 mask = (curv >= low) & (curv < high)

#             if mask.any():
#                 pred_bin = predictions[mask].view(-1)
#                 label_bin = labels[mask].view(-1)
#                 if pred_bin.numel() == 0:
#                     continue
#                 acc = (pred_bin == label_bin).float().mean().item()

#                 # compute IoU per class
#                 confusion = torch.zeros(self.num_classes, self.num_classes, device=predictions.device)
#                 for c_pred in range(self.num_classes):
#                     for c_true in range(self.num_classes):
#                         confusion[c_pred, c_true] = ((pred_bin == c_pred) & (label_bin == c_true)).sum()

#                 ious = []
#                 for c in range(self.num_classes):
#                     inter = confusion[c, c].item()
#                     union = (confusion[c, :].sum() + confusion[:, c].sum() - confusion[c, c]).item()
#                     iou = float(inter / union) if union > 0 else 0.0
#                     ious.append(iou)

#                 results[f'accuracy_curvature_bin_{i}'] = acc
#                 results[f'mIoU_curvature_bin_{i}'] = float(np.mean(ious))

#         return results




# import torch
# import torch.nn.functional as F
# import numpy as np
# from scipy import ndimage
# from typing import Dict, Tuple, Optional


# class BoundaryAwareMetrics:
#     """
#     Implements boundary-aware evaluation metrics for 3D point cloud segmentation.
#     Critical for demonstrating improvements at class boundaries.
#     """
    
#     def __init__(self, num_classes: int, boundary_width: float = 0.1):
#         self.num_classes = num_classes
#         self.boundary_width = boundary_width
#         self.reset()
    
#     def reset(self):
#         """Reset all metric accumulators"""
#         self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
#         self.boundary_confusion_matrix = np.zeros((self.num_classes, self.num_classes))
#         self.boundary_points_count = 0
#         self.total_points_count = 0
        
#     def compute_boundary_mask(self, xyz: torch.Tensor, labels: torch.Tensor, k: int = 16) -> torch.Tensor:
#         """
#         Identify boundary points based on label discontinuities in local neighborhoods.
        
#         Args:
#             xyz: Point coordinates (B, N, 3)
#             labels: Point labels (B, N)
#             k: Number of neighbors to consider
            
#         Returns:
#             boundary_mask: Boolean mask indicating boundary points (B, N)
#         """
#         B, N, _ = xyz.shape
#         device = xyz.device
        
#         # Compute pairwise distances
#         xyz_expanded = xyz.unsqueeze(2)  # (B, N, 1, 3)
#         xyz_transposed = xyz.unsqueeze(1)  # (B, 1, N, 3)
#         distances = torch.norm(xyz_expanded - xyz_transposed, dim=-1)  # (B, N, N)
        
#         # Get k-nearest neighbors
#         _, nn_indices = torch.topk(distances, k=k+1, dim=-1, largest=False)
#         nn_indices = nn_indices[:, :, 1:]  # Exclude self
        
#         # Gather neighbor labels
#         batch_indices = torch.arange(B, device=device).view(B, 1, 1).expand(B, N, k)
#         neighbor_labels = labels[batch_indices, nn_indices]  # (B, N, k)
        
#         # Check if any neighbor has different label
#         center_labels = labels.unsqueeze(-1).expand_as(neighbor_labels)
#         boundary_mask = (neighbor_labels != center_labels).any(dim=-1)
        
#         return boundary_mask
    
#     def compute_curvature_weights(self, xyz: torch.Tensor, k: int = 16) -> torch.Tensor:
#         """
#         Compute local curvature as importance weights for evaluation.
#         High curvature regions are more challenging and important.
        
#         Args:
#             xyz: Point coordinates (B, N, 3)
#             k: Number of neighbors for curvature estimation
            
#         Returns:
#             curvature_weights: Normalized weights based on local curvature (B, N)
#         """
#         B, N, _ = xyz.shape
#         device = xyz.device
        
#         # Get k-nearest neighbors
#         xyz_expanded = xyz.unsqueeze(2)
#         xyz_transposed = xyz.unsqueeze(1)
#         distances = torch.norm(xyz_expanded - xyz_transposed, dim=-1)
#         _, nn_indices = torch.topk(distances, k=k+1, dim=-1, largest=False)
#         nn_indices = nn_indices[:, :, 1:]
        
#         # Gather neighbor coordinates
#         batch_indices = torch.arange(B, device=device).view(B, 1, 1).expand(B, N, k)
#         neighbor_xyz = xyz[batch_indices, nn_indices]  # (B, N, k, 3)
        
#         # Compute local covariance
#         centered_neighbors = neighbor_xyz - xyz.unsqueeze(2)  # (B, N, k, 3)
#         cov = torch.matmul(centered_neighbors.transpose(-1, -2), centered_neighbors) / k  # (B, N, 3, 3)
        
#         # Compute eigenvalues as curvature measure
#         eigenvalues = torch.linalg.eigvalsh(cov)  # (B, N, 3)
        
#         # Curvature = ratio of smallest to sum of eigenvalues (surface variation)
#         curvature = eigenvalues[:, :, 0] / (eigenvalues.sum(dim=-1) + 1e-8)
        
#         # Normalize to [0, 1]
#         curvature_weights = (curvature - curvature.min()) / (curvature.max() - curvature.min() + 1e-8)
        
#         return curvature_weights
    
#     def update(self, predictions: torch.Tensor, labels: torch.Tensor, xyz: torch.Tensor):
#         """
#         Update metrics with new predictions.
        
#         Args:
#             predictions: Predicted labels (B, N) or logits (B, N, C)
#             labels: Ground truth labels (B, N)
#             xyz: Point coordinates (B, N, 3)
#         """
#         if predictions.dim() == 3:
#             predictions = predictions.argmax(dim=-1)
            
#         # Compute boundary mask
#         boundary_mask = self.compute_boundary_mask(xyz, labels)
        
#         # Update confusion matrices
#         for b in range(predictions.shape[0]):
#             pred_b = predictions[b].cpu().numpy()
#             label_b = labels[b].cpu().numpy()
#             boundary_b = boundary_mask[b].cpu().numpy()
            
#             # Overall confusion matrix
#             for i in range(self.num_classes):
#                 for j in range(self.num_classes):
#                     self.confusion_matrix[i, j] += ((pred_b == i) & (label_b == j)).sum()
            
#             # Boundary confusion matrix
#             boundary_pred = pred_b[boundary_b]
#             boundary_label = label_b[boundary_b]
#             for i in range(self.num_classes):
#                 for j in range(self.num_classes):
#                     self.boundary_confusion_matrix[i, j] += ((boundary_pred == i) & (boundary_label == j)).sum()
            
#             self.boundary_points_count += boundary_b.sum()
#             self.total_points_count += len(pred_b)
    
#     def compute_metrics(self) -> Dict[str, float]:
#         """Compute all evaluation metrics"""
#         metrics = {}
        
#         # Standard metrics
#         metrics.update(self._compute_standard_metrics())
        
#         # Boundary-specific metrics
#         metrics.update(self._compute_boundary_metrics())
        
#         # Boundary improvement ratio
#         if metrics['mIoU'] > 0:
#             metrics['boundary_improvement_ratio'] = metrics['boundary_mIoU'] / metrics['mIoU']
        
#         return metrics
    
#     def _compute_standard_metrics(self) -> Dict[str, float]:
#         """Compute standard segmentation metrics"""
#         metrics = {}
        
#         # Overall accuracy
#         correct = np.diag(self.confusion_matrix).sum()
#         total = self.confusion_matrix.sum()
#         metrics['overall_accuracy'] = correct / total if total > 0 else 0
        
#         # Per-class IoU
#         iou_per_class = []
#         for i in range(self.num_classes):
#             intersection = self.confusion_matrix[i, i]
#             union = self.confusion_matrix[i, :].sum() + self.confusion_matrix[:, i].sum() - intersection
#             iou = intersection / union if union > 0 else 0
#             iou_per_class.append(iou)
#             metrics[f'IoU_class_{i}'] = iou
        
#         # Mean IoU
#         metrics['mIoU'] = np.mean(iou_per_class)
        
#         return metrics
    
#     def _compute_boundary_metrics(self) -> Dict[str, float]:
#         """Compute boundary-specific metrics"""
#         metrics = {}
        
#         # Boundary accuracy
#         boundary_correct = np.diag(self.boundary_confusion_matrix).sum()
#         metrics['boundary_accuracy'] = boundary_correct / self.boundary_points_count if self.boundary_points_count > 0 else 0
        
#         # Boundary IoU per class
#         boundary_iou_per_class = []
#         for i in range(self.num_classes):
#             intersection = self.boundary_confusion_matrix[i, i]
#             union = self.boundary_confusion_matrix[i, :].sum() + self.boundary_confusion_matrix[:, i].sum() - intersection
#             iou = intersection / union if union > 0 else 0
#             boundary_iou_per_class.append(iou)
#             metrics[f'boundary_IoU_class_{i}'] = iou
        
#         # Mean boundary IoU
#         metrics['boundary_mIoU'] = np.mean(boundary_iou_per_class)
        
#         # Boundary F1 scores
#         for i in range(self.num_classes):
#             tp = self.boundary_confusion_matrix[i, i]
#             fp = self.boundary_confusion_matrix[i, :].sum() - tp
#             fn = self.boundary_confusion_matrix[:, i].sum() - tp
            
#             precision = tp / (tp + fp) if (tp + fp) > 0 else 0
#             recall = tp / (tp + fn) if (tp + fn) > 0 else 0
#             f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
#             metrics[f'boundary_F1_class_{i}'] = f1
        
#         # Mean boundary F1
#         boundary_f1_scores = [metrics[f'boundary_F1_class_{i}'] for i in range(self.num_classes)]
#         metrics['boundary_mF1'] = np.mean(boundary_f1_scores)
        
#         # Boundary point percentage
#         metrics['boundary_point_percentage'] = self.boundary_points_count / self.total_points_count if self.total_points_count > 0 else 0
        
#         return metrics


# class CurvatureAwareEvaluator:
#     """
#     Evaluates model performance with respect to surface curvature.
#     Essential for demonstrating that GAS preserves high-curvature regions.
#     """
    
#     def __init__(self, num_classes: int, curvature_bins: int = 5):
#         self.num_classes = num_classes
#         self.curvature_bins = curvature_bins
#         self.reset()
    
#     def reset(self):
#         """Reset evaluation state"""
#         self.bin_accuracies = {i: [] for i in range(self.curvature_bins)}
#         self.bin_ious = {i: [] for i in range(self.curvature_bins)}
        
#     def evaluate(self, predictions: torch.Tensor, labels: torch.Tensor, xyz: torch.Tensor) -> Dict[str, float]:
#         """
#         Evaluate predictions stratified by curvature levels.
        
#         Args:
#             predictions: Model predictions
#             labels: Ground truth
#             xyz: Point coordinates
            
#         Returns:
#             Metrics stratified by curvature
#         """
#         if predictions.dim() == 3:
#             predictions = predictions.argmax(dim=-1)
        
#         # Compute curvature for each point
#         metrics_computer = BoundaryAwareMetrics(self.num_classes)
#         curvatures = metrics_computer.compute_curvature_weights(xyz)
        
#         # Bin points by curvature
#         curvature_thresholds = torch.linspace(0, 1, self.curvature_bins + 1)
        
#         results = {}
#         for i in range(self.curvature_bins):
#             low, high = curvature_thresholds[i], curvature_thresholds[i + 1]
#             if i == self.curvature_bins - 1:
#                 mask = (curvatures >= low) & (curvatures <= high)
#             else:
#                 mask = (curvatures >= low) & (curvatures < high)
            
#             if mask.any():
#                 bin_pred = predictions[mask]
#                 bin_label = labels[mask]
                
#                 # Compute accuracy for this bin
#                 accuracy = (bin_pred == bin_label).float().mean().item()
#                 self.bin_accuracies[i].append(accuracy)
                
#                 # Compute IoU for this bin
#                 confusion = torch.zeros(self.num_classes, self.num_classes)
#                 for c_pred in range(self.num_classes):
#                     for c_true in range(self.num_classes):
#                         confusion[c_pred, c_true] = ((bin_pred == c_pred) & (bin_label == c_true)).sum()
                
#                 ious = []
#                 for c in range(self.num_classes):
#                     intersection = confusion[c, c]
#                     union = confusion[c, :].sum() + confusion[:, c].sum() - intersection
#                     iou = intersection / union if union > 0 else 0
#                     ious.append(iou.item())
                
#                 self.bin_ious[i].append(np.mean(ious))
        
#         # Aggregate results
#         for i in range(self.curvature_bins):
#             if self.bin_accuracies[i]:
#                 results[f'accuracy_curvature_bin_{i}'] = np.mean(self.bin_accuracies[i])
#                 results[f'mIoU_curvature_bin_{i}'] = np.mean(self.bin_ious[i])
        
#         return results