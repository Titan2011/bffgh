import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import os


class BoundaryVisualization:
    """
    Visualization tools for analyzing boundary preservation and errors.
    Critical for demonstrating GAS effectiveness in publications.
    """
    
    def __init__(self, save_dir: str = 'visualizations'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def visualize_boundary_errors(self, 
                                  xyz: torch.Tensor, 
                                  predictions: torch.Tensor, 
                                  labels: torch.Tensor,
                                  scene_name: str,
                                  num_classes: int,
                                  knn_fn: callable):
        """
        Create comprehensive boundary error visualizations.
        
        Args:
            xyz: Point coordinates (N, 3)
            predictions: Predicted labels (N,)
            labels: Ground truth labels (N,)
            scene_name: Name for saving
            num_classes: Number of classes
        """
        # Convert to numpy
        xyz = xyz.cpu().numpy()
        predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy()
        
        # Identify boundary points
        from boundary_metrics import BoundaryAwareMetrics
        metrics = BoundaryAwareMetrics(num_classes)
        boundary_mask = metrics._compute_boundary_mask_band(
            xyz.unsqueeze(0),
            labels.unsqueeze(0),
            knn_fn=knn_fn,
            k=metrics.k_boundary,
            boundary_width=metrics.boundary_width
        )[0].cpu().numpy()
        
        # Identify errors
        errors = predictions != labels
        boundary_errors = errors & boundary_mask
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 5))
        
        # 1. Ground truth
        ax1 = fig.add_subplot(141, projection='3d')
        scatter1 = ax1.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], 
                              c=labels, cmap='tab20', s=0.5, alpha=0.8)
        ax1.set_title('Ground Truth')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # 2. Predictions
        ax2 = fig.add_subplot(142, projection='3d')
        scatter2 = ax2.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], 
                              c=predictions, cmap='tab20', s=0.5, alpha=0.8)
        ax2.set_title('Predictions')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        
        # 3. All errors (red) vs correct (gray)
        ax3 = fig.add_subplot(143, projection='3d')
        colors = np.where(errors, 'red', 'lightgray')
        ax3.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], 
                   c=colors, s=0.5, alpha=0.8)
        ax3.set_title(f'All Errors (Error rate: {errors.mean():.2%})')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        
        # 4. Boundary errors highlighted
        ax4 = fig.add_subplot(144, projection='3d')
        # Plot all points in light gray
        ax4.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], 
                   c='lightgray', s=0.1, alpha=0.3)
        # Highlight boundary points
        boundary_xyz = xyz[boundary_mask]
        boundary_colors = np.where(boundary_errors[boundary_mask], 'red', 'green')
        ax4.scatter(boundary_xyz[:, 0], boundary_xyz[:, 1], boundary_xyz[:, 2],
                   c=boundary_colors, s=2, alpha=1.0)
        
        boundary_error_rate = boundary_errors[boundary_mask].mean() if boundary_mask.any() else 0
        ax4.set_title(f'Boundary Errors (Error rate: {boundary_error_rate:.2%})')
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        ax4.set_zlabel('Z')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'{scene_name}_boundary_errors.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def visualize_sampling_distribution(self,
                                       xyz: torch.Tensor,
                                       sampled_indices: torch.Tensor,
                                       aux_info: Dict[str, torch.Tensor],
                                       scene_name: str,
                                       layer_idx: int):
        """
        Visualize how GAS distributes sampling across the point cloud.
        
        Args:
            xyz: Original point coordinates (N, 3)
            sampled_indices: Indices of sampled points (k,)
            aux_info: Auxiliary information from GAS
            scene_name: Name for saving
            layer_idx: Which layer this sampling is from
        """
        xyz = xyz.cpu().numpy()
        sampled_indices = sampled_indices.cpu().numpy()
        
        # Extract scores
        curvature_scores = aux_info['curvature_scores'].cpu().numpy()
        boundary_scores = aux_info['boundary_scores'].cpu().numpy()
        importance_scores = aux_info['importance_scores'].cpu().numpy()
        
        fig = plt.figure(figsize=(20, 10))
        
        # 1. Curvature scores
        ax1 = fig.add_subplot(241, projection='3d')
        scatter1 = ax1.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                              c=curvature_scores, cmap='viridis', s=0.5, alpha=0.8)
        plt.colorbar(scatter1, ax=ax1, label='Curvature Score')
        ax1.set_title('Curvature Scores')
        
        # 2. Boundary scores
        ax2 = fig.add_subplot(242, projection='3d')
        scatter2 = ax2.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                              c=boundary_scores, cmap='plasma', s=0.5, alpha=0.8)
        plt.colorbar(scatter2, ax=ax2, label='Boundary Score')
        ax2.set_title('Boundary Scores')
        
        # 3. Combined importance scores
        ax3 = fig.add_subplot(243, projection='3d')
        scatter3 = ax3.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                              c=importance_scores, cmap='hot', s=0.5, alpha=0.8)
        plt.colorbar(scatter3, ax=ax3, label='Importance Score')
        ax3.set_title('Combined Importance Scores')
        
        # 4. Sampled points
        ax4 = fig.add_subplot(244, projection='3d')
        # Plot all points in gray
        ax4.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                   c='lightgray', s=0.1, alpha=0.3)
        # Highlight sampled points
        sampled_xyz = xyz[sampled_indices]
        ax4.scatter(sampled_xyz[:, 0], sampled_xyz[:, 1], sampled_xyz[:, 2],
                   c='red', s=2, alpha=1.0)
        ax4.set_title(f'Sampled Points ({len(sampled_indices)}/{len(xyz)})')
        
        # 5-8. Score distributions
        ax5 = fig.add_subplot(245)
        ax5.hist(curvature_scores, bins=50, alpha=0.7, color='green')
        ax5.axvline(curvature_scores[sampled_indices].mean(), color='red', linestyle='--', 
                   label=f'Sampled mean: {curvature_scores[sampled_indices].mean():.3f}')
        ax5.axvline(curvature_scores.mean(), color='black', linestyle='--',
                   label=f'Overall mean: {curvature_scores.mean():.3f}')
        ax5.set_xlabel('Curvature Score')
        ax5.set_ylabel('Count')
        ax5.set_title('Curvature Score Distribution')
        ax5.legend()
        
        ax6 = fig.add_subplot(246)
        ax6.hist(boundary_scores, bins=50, alpha=0.7, color='blue')
        ax6.axvline(boundary_scores[sampled_indices].mean(), color='red', linestyle='--',
                   label=f'Sampled mean: {boundary_scores[sampled_indices].mean():.3f}')
        ax6.axvline(boundary_scores.mean(), color='black', linestyle='--',
                   label=f'Overall mean: {boundary_scores.mean():.3f}')
        ax6.set_xlabel('Boundary Score')
        ax6.set_ylabel('Count')
        ax6.set_title('Boundary Score Distribution')
        ax6.legend()
        
        ax7 = fig.add_subplot(247)
        ax7.hist(importance_scores, bins=50, alpha=0.7, color='orange')
        ax7.axvline(importance_scores[sampled_indices].mean(), color='red', linestyle='--',
                   label=f'Sampled mean: {importance_scores[sampled_indices].mean():.3f}')
        ax7.axvline(importance_scores.mean(), color='black', linestyle='--',
                   label=f'Overall mean: {importance_scores.mean():.3f}')
        ax7.set_xlabel('Importance Score')
        ax7.set_ylabel('Count')
        ax7.set_title('Importance Score Distribution')
        ax7.legend()
        
        # 8. Correlation plot
        ax8 = fig.add_subplot(248)
        ax8.scatter(curvature_scores, boundary_scores, s=1, alpha=0.5, c='gray', label='All points')
        ax8.scatter(curvature_scores[sampled_indices], boundary_scores[sampled_indices],
                   s=5, alpha=0.8, c='red', label='Sampled points')
        ax8.set_xlabel('Curvature Score')
        ax8.set_ylabel('Boundary Score')
        ax8.set_title('Score Correlation')
        ax8.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'{scene_name}_layer{layer_idx}_sampling.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_metrics_comparison(self, results: Dict[str, Dict[str, float]], save_name: str):
        """
        Create publication-quality plots comparing different sampling strategies.
        
        Args:
            results: Dictionary mapping strategy names to their metrics
            save_name: Name for saving the plot
        """
        # Extract key metrics for comparison
        strategies = list(results.keys())
        metrics_to_plot = [
            'mIoU', 'boundary_mIoU', 'boundary_mF1',
            'mIoU_curvature_bin_0', 'mIoU_curvature_bin_4'  # Low and high curvature
        ]
        
        metric_names = [
            'Overall mIoU', 'Boundary mIoU', 'Boundary mF1',
            'Low Curvature mIoU', 'High Curvature mIoU'
        ]
        
        # Create figure
        fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(20, 4))
        
        for idx, (metric, name) in enumerate(zip(metrics_to_plot, metric_names)):
            ax = axes[idx]
            
            values = []
            for strategy in strategies:
                if metric in results[strategy]:
                    values.append(results[strategy][metric] * 100)  # Convert to percentage
                else:
                    values.append(0)
            
            # Create bar plot
            bars = ax.bar(strategies, values)
            
            # Color GAS differently
            for i, (strategy, bar) in enumerate(zip(strategies, bars)):
                if strategy == 'gas':
                    bar.set_color('red')
                else:
                    bar.set_color('gray')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.1f}%', ha='center', va='bottom', fontsize=10)
            
            ax.set_ylabel('Performance (%)')
            ax.set_title(name)
            ax.set_ylim(0, max(values) * 1.1)
            
            # Rotate x labels
            ax.set_xticklabels(strategies, rotation=45, ha='right')
        
        plt.suptitle('Sampling Strategy Comparison', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'{save_name}_metrics_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create improvement heatmap
        self._plot_improvement_heatmap(results, save_name)
        
    def _plot_improvement_heatmap(self, results: Dict[str, Dict[str, float]], save_name: str):
        """Create heatmap showing improvements of GAS over baselines."""
        
        if 'gas' not in results:
            return
            
        baselines = ['random', 'grid', 'fps']
        metrics = ['mIoU', 'boundary_mIoU', 'boundary_mF1']
        
        # Compute improvements
        improvements = np.zeros((len(metrics), len(baselines)))
        
        for i, metric in enumerate(metrics):
            for j, baseline in enumerate(baselines):
                if baseline in results and metric in results[baseline] and metric in results['gas']:
                    baseline_val = results[baseline][metric]
                    gas_val = results['gas'][metric]
                    if baseline_val > 0:
                        improvements[i, j] = (gas_val - baseline_val) / baseline_val * 100
        
        # Create heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(improvements, 
                   xticklabels=baselines,
                   yticklabels=['Overall mIoU', 'Boundary mIoU', 'Boundary mF1'],
                   annot=True, 
                   fmt='.1f',
                   cmap='RdYlGn',
                   center=0,
                   cbar_kws={'label': 'Improvement (%)'},
                   annot_kws={'fontsize': 12})
        
        plt.title('GAS Improvement over Baseline Sampling Strategies', fontsize=14)
        plt.xlabel('Baseline Strategy', fontsize=12)
        plt.ylabel('Metric', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'{save_name}_improvement_heatmap.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()