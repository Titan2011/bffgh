"""
SOTA Model Comparisons for A* Publication
Implements baseline models for fair comparison with GAS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
from abc import ABC, abstractmethod
from boundary_metrics import BoundaryAwareMetrics
from tqdm import tqdm


class BaselineModel(ABC):
    """Abstract base class for baseline models."""
    
    @abstractmethod
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pass
    
    @abstractmethod
    def get_sampling_method(self) -> str:
        pass


class RandomSamplingRandLA(BaselineModel):
    """RandLA-Net with random sampling (baseline)."""
    
    def __init__(self, config):
        from RandLANet_fixed import Network
        self.model = Network(config)
        self.sampling_method = "random"
        
    def forward(self, batch):
        # Override sampling in the model to use random
        return self.model(batch)
    
    def get_sampling_method(self):
        return self.sampling_method


class GridSamplingRandLA(BaselineModel):
    """RandLA-Net with grid-based voxel sampling."""
    
    def __init__(self, config):
        from RandLANet_fixed import Network
        self.model = Network(config)
        self.sampling_method = "grid"
        self.voxel_size = 0.06  # Adjust based on dataset
        
    def voxel_sampling(self, xyz: torch.Tensor, features: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Grid-based voxel sampling."""
        B, N, _ = xyz.shape
        
        # Voxelize the point cloud
        voxel_coords = torch.floor(xyz / self.voxel_size).long()
        
        # Get unique voxels
        sampled_xyz = []
        sampled_features = []
        sampled_indices = []
        
        for b in range(B):
            # Find unique voxels
            unique_voxels, inverse_indices = torch.unique(
                voxel_coords[b], dim=0, return_inverse=True
            )
            
            # Average points in each voxel
            voxel_features = torch.zeros(len(unique_voxels), features.shape[-1], device=features.device)
            voxel_xyz = torch.zeros(len(unique_voxels), 3, device=xyz.device)
            
            for i in range(len(unique_voxels)):
                mask = inverse_indices == i
                voxel_features[i] = features[b, mask].mean(dim=0)
                voxel_xyz[i] = xyz[b, mask].mean(dim=0)
            
            # Sample k voxels
            if len(unique_voxels) > k:
                perm = torch.randperm(len(unique_voxels))[:k]
                voxel_features = voxel_features[perm]
                voxel_xyz = voxel_xyz[perm]
            
            sampled_xyz.append(voxel_xyz)
            sampled_features.append(voxel_features)
        
        return torch.stack(sampled_xyz), torch.stack(sampled_features), None
    
    def forward(self, batch):
        # Override sampling to use grid
        return self.model(batch)
    
    def get_sampling_method(self):
        return self.sampling_method


class FPSSamplingRandLA(BaselineModel):
    """RandLA-Net with Farthest Point Sampling."""
    
    def __init__(self, config):
        from RandLANet_fixed import Network
        self.model = Network(config)
        self.sampling_method = "fps"
        
    def farthest_point_sampling(self, xyz: torch.Tensor, k: int) -> torch.Tensor:
        """
        Farthest Point Sampling implementation.
        
        Args:
            xyz: Point coordinates (B, N, 3)
            k: Number of points to sample
            
        Returns:
            Indices of sampled points (B, k)
        """
        B, N, _ = xyz.shape
        device = xyz.device
        
        centroids = torch.zeros(B, k, dtype=torch.long, device=device)
        distance = torch.ones(B, N, device=device) * 1e10
        
        # Random initialization
        farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
        
        for i in range(k):
            centroids[:, i] = farthest
            
            # Update distances
            centroid_xyz = xyz[torch.arange(B), farthest].unsqueeze(1)  # (B, 1, 3)
            dist = torch.sum((xyz - centroid_xyz) ** 2, dim=-1)  # (B, N)
            
            mask = dist < distance
            distance[mask] = dist[mask]
            
            # Select farthest point
            farthest = torch.max(distance, dim=1)[1]
        
        return centroids
    
    def forward(self, batch):
        # Override sampling to use FPS
        return self.model(batch)
    
    def get_sampling_method(self):
        return self.sampling_method


class SOTAComparison:
    """
    Comprehensive comparison framework for A* publication.
    Ensures fair comparison across all methods.
    """
    
    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = device
        self.num_classes = config.num_classes
        
        # Initialize all baseline models
        self.models = {
            'Random': RandomSamplingRandLA(config),
            'Grid': GridSamplingRandLA(config),
            'FPS': FPSSamplingRandLA(config),
            'GAS (Ours)': None  # Will be loaded separately
        }
        
        # Initialize metrics
        self.boundary_metrics = BoundaryAwareMetrics(self.num_classes)
        
    def load_pretrained_models(self, model_paths: Dict[str, str]):
        """Load pretrained weights for each model."""
        for name, path in model_paths.items():
            if name in self.models and self.models[name] is not None:
                checkpoint = torch.load(path, map_location=self.device)
                self.models[name].model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded {name} from {path}")
    
    def evaluate_single_model(self, model_name: str, model, test_loader) -> Dict[str, float]:
        """Evaluate a single model with comprehensive metrics."""
        print(f"\nEvaluating {model_name}...")
        
        model.eval()
        self.boundary_metrics.reset()
        
        all_predictions = []
        all_labels = []
        all_xyz = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Testing {model_name}"):
                # Move to device
                for key in batch:
                    if isinstance(batch[key], list):
                        for j in range(len(batch[key])):
                            batch[key][j] = batch[key][j].to(self.device)
                    else:
                        batch[key] = batch[key].to(self.device)
                
                # Forward pass
                outputs = model(batch)
                predictions = outputs['logits'].argmax(dim=-1)
                
                # Store for analysis
                all_predictions.append(predictions)
                all_labels.append(batch['labels'])
                all_xyz.append(batch['xyz'][0])
                
                # Update metrics
                self.boundary_metrics.update(
                    outputs['logits'], 
                    batch['labels'], 
                    batch['xyz'][0]
                )
        
        # Compute all metrics
        metrics = self.boundary_metrics.compute_metrics()
        
        # Add additional analysis
        metrics.update(self._compute_additional_metrics(
            torch.cat(all_predictions),
            torch.cat(all_labels),
            torch.cat(all_xyz)
        ))
        
        return metrics
    
    def _compute_additional_metrics(self, predictions, labels, xyz):
        """Compute additional metrics for comprehensive evaluation."""
        metrics = {}
        
        # Per-class IoU
        for c in range(self.num_classes):
            mask = labels == c
            if mask.any():
                intersection = ((predictions == c) & mask).sum().float()
                union = ((predictions == c) | mask).sum().float()
                metrics[f'class_{c}_iou'] = (intersection / union).item()
        
        # Compute metrics by point density
        # This helps understand performance in sparse vs dense regions
        
        return metrics
    
    def run_comprehensive_comparison(self, test_loader, num_runs: int = 5) -> Dict[str, Dict[str, List[float]]]:
        """
        Run comparison with multiple seeds for statistical significance.
        
        Returns:
            Dictionary mapping model names to their metrics across runs
        """
        results = {name: {} for name in self.models.keys()}
        
        for run in range(num_runs):
            print(f"\n{'='*50}")
            print(f"Run {run + 1}/{num_runs}")
            print(f"{'='*50}")
            
            # Set random seed for reproducibility
            torch.manual_seed(42 + run)
            np.random.seed(42 + run)
            
            for model_name, model in self.models.items():
                if model is None:
                    continue
                    
                metrics = self.evaluate_single_model(model_name, model.model, test_loader)
                
                # Store results
                for metric_name, value in metrics.items():
                    if metric_name not in results[model_name]:
                        results[model_name][metric_name] = []
                    results[model_name][metric_name].append(value)
        
        return results
    
    def analyze_results(self, results: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Analyze results with statistical significance testing.
        
        Returns:
            Dictionary with mean, std, and statistical tests
        """
        from scipy import stats
        
        analysis = {}
        
        # Compute statistics for each model
        for model_name, model_results in results.items():
            analysis[model_name] = {}
            
            for metric_name, values in model_results.items():
                analysis[model_name][metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'values': values
                }
        
        # Statistical significance testing (GAS vs others)
        if 'GAS (Ours)' in results:
            gas_results = results['GAS (Ours)']
            
            for model_name in results:
                if model_name == 'GAS (Ours)':
                    continue
                    
                print(f"\n{model_name} vs GAS (Ours):")
                
                for metric in ['mIoU', 'boundary_mIoU', 'boundary_mF1']:
                    if metric in gas_results and metric in results[model_name]:
                        gas_values = gas_results[metric]
                        baseline_values = results[model_name][metric]
                        
                        # Paired t-test
                        t_stat, p_value = stats.ttest_rel(gas_values, baseline_values)
                        
                        improvement = (np.mean(gas_values) - np.mean(baseline_values)) / np.mean(baseline_values) * 100
                        
                        print(f"  {metric}: {improvement:+.1f}% (p={p_value:.4f})")
                        
                        if p_value < 0.05:
                            print(f"    ✓ Statistically significant")
                        else:
                            print(f"    ✗ Not statistically significant")
        
        return analysis
    
    def generate_latex_table(self, analysis: Dict[str, Dict[str, Dict[str, float]]]):
        """Generate LaTeX table for paper."""
        
        metrics_to_show = ['mIoU', 'boundary_mIoU', 'boundary_mF1', 'mIoU_curvature_bin_4']
        metric_names = ['Overall mIoU', 'Boundary mIoU', 'Boundary F1', 'High-Curv mIoU']
        
        print("\n% LaTeX Table for Paper")
        print("\\begin{table}[t]")
        print("\\centering")
        print("\\caption{Comparison of sampling strategies on Semantic3D. Best results in \\textbf{bold}.}")
        print("\\label{tab:main_results}")
        print("\\begin{tabular}{l" + "c" * len(metrics_to_show) + "}")
        print("\\toprule")
        print("Method & " + " & ".join(metric_names) + " \\\\")
        print("\\midrule")
        
        # Find best values for each metric
        best_values = {}
        for metric in metrics_to_show:
            best_values[metric] = max(
                analysis[model][metric]['mean'] 
                for model in analysis 
                if metric in analysis[model]
            )
        
        # Print results for each model
        for model_name in ['Random', 'Grid', 'FPS', 'GAS (Ours)']:
            if model_name not in analysis:
                continue
                
            row = model_name
            
            for metric in metrics_to_show:
                if metric in analysis[model_name]:
                    mean = analysis[model_name][metric]['mean'] * 100  # Convert to percentage
                    std = analysis[model_name][metric]['std'] * 100
                    
                    # Bold if best
                    if analysis[model_name][metric]['mean'] == best_values[metric]:
                        row += f" & \\textbf{{{mean:.1f} ± {std:.1f}}}"
                    else:
                        row += f" & {mean:.1f} ± {std:.1f}"
                else:
                    row += " & -"
            
            row += " \\\\"
            print(row)
        
        print("\\bottomrule")
        print("\\end{tabular}")
        print("\\end{table}")


def main():
    """Example usage for SOTA comparison."""
    from helper_tool import ConfigSemantic3D as cfg
    from main_Semantic3D import get_dataloader
    
    # Initialize comparison framework
    comparison = SOTAComparison(cfg)
    
    # Load test data
    test_loader, _ = get_dataloader('test')
    
    # Load pretrained models
    model_paths = {
        'Random': 'checkpoints/randla_random_best.pth',
        'Grid': 'checkpoints/randla_grid_best.pth',
        'FPS': 'checkpoints/randla_fps_best.pth',
        'GAS (Ours)': 'checkpoints/randla_gas_best.pth'
    }
    comparison.load_pretrained_models(model_paths)
    
    # Run comprehensive comparison
    results = comparison.run_comprehensive_comparison(test_loader, num_runs=5)
    
    # Analyze results
    analysis = comparison.analyze_results(results)
    
    # Generate LaTeX table
    comparison.generate_latex_table(analysis)
    
    # Save results
    import pickle
    with open('sota_comparison_results.pkl', 'wb') as f:
        pickle.dump({
            'results': results,
            'analysis': analysis
        }, f)


if __name__ == '__main__':
    main()