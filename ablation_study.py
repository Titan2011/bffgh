import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod


class SamplingStrategy(ABC):
    """Abstract base class for different sampling strategies."""
    
    @abstractmethod
    def sample(self, xyz: torch.Tensor, features: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample k points from the input.
        
        Args:
            xyz: Point coordinates (B, N, 3)
            features: Point features (B, N, C)
            k: Number of points to sample
            
        Returns:
            sampled_features: (B, k, C)
            sampled_xyz: (B, k, 3)
            indices: (B, k)
        """
        pass


class RandomSampling(SamplingStrategy):
    """Baseline: Uniform random sampling."""
    
    def sample(self, xyz: torch.Tensor, features: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, _ = xyz.shape
        
        # Random indices
        indices = torch.multinomial(torch.ones(B, N, device=xyz.device), k, replacement=False)
        
        # Gather sampled points
        batch_indices = torch.arange(B, device=xyz.device).view(B, 1).expand(B, k)
        sampled_xyz = xyz[batch_indices, indices]
        sampled_features = features[batch_indices, indices]
        
        return sampled_features, sampled_xyz, indices


class GridSampling(SamplingStrategy):
    """Baseline: Grid-based voxel sampling."""
    
    def __init__(self, voxel_size: Optional[float] = None):
        self.voxel_size = voxel_size
    
    def sample(self, xyz: torch.Tensor, features: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, _ = xyz.shape
        device = xyz.device
        
        sampled_features_list = []
        sampled_xyz_list = []
        indices_list = []
        
        for b in range(B):
            xyz_b = xyz[b]
            features_b = features[b]
            
            # Compute voxel size if not provided
            if self.voxel_size is None:
                # Adaptive voxel size to get approximately k points
                volume = (xyz_b.max(0)[0] - xyz_b.min(0)[0]).prod()
                voxel_size = (volume / k) ** (1/3)
            else:
                voxel_size = self.voxel_size
            
            # Voxelize
            voxel_coords = torch.floor(xyz_b / voxel_size).long()
            
            # Get unique voxels and select one point per voxel
            unique_voxels, inverse_indices = torch.unique(voxel_coords, dim=0, return_inverse=True)
            
            selected_indices = []
            for i in range(len(unique_voxels)):
                voxel_points = torch.where(inverse_indices == i)[0]
                # Select center-most point in voxel
                voxel_xyz = xyz_b[voxel_points]
                voxel_center = voxel_xyz.mean(0)
                distances = torch.norm(voxel_xyz - voxel_center, dim=1)
                selected_idx = voxel_points[distances.argmin()]
                selected_indices.append(selected_idx)
            
            selected_indices = torch.tensor(selected_indices, device=device)
            
            # If we have too many points, randomly subsample
            if len(selected_indices) > k:
                perm = torch.randperm(len(selected_indices), device=device)[:k]
                selected_indices = selected_indices[perm]
            # If too few, add random points
            elif len(selected_indices) < k:
                remaining = k - len(selected_indices)
                mask = torch.ones(N, dtype=torch.bool, device=device)
                mask[selected_indices] = False
                additional_indices = torch.where(mask)[0]
                perm = torch.randperm(len(additional_indices), device=device)[:remaining]
                selected_indices = torch.cat([selected_indices, additional_indices[perm]])
            
            sampled_xyz_list.append(xyz_b[selected_indices])
            sampled_features_list.append(features_b[selected_indices])
            indices_list.append(selected_indices)
        
        # Stack batch
        sampled_xyz = torch.stack(sampled_xyz_list)
        sampled_features = torch.stack(sampled_features_list)
        indices = torch.stack(indices_list)
        
        return sampled_features, sampled_xyz, indices


class FarthestPointSampling(SamplingStrategy):
    """Baseline: Farthest Point Sampling (FPS)."""
    
    def sample(self, xyz: torch.Tensor, features: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, _ = xyz.shape
        device = xyz.device
        
        sampled_features_list = []
        sampled_xyz_list = []
        indices_list = []
        
        for b in range(B):
            xyz_b = xyz[b]
            features_b = features[b]
            
            # Initialize with random point
            indices = torch.zeros(k, dtype=torch.long, device=device)
            distances = torch.ones(N, device=device) * 1e10
            
            # Random starting point
            indices[0] = torch.randint(0, N, (1,), device=device)
            
            for i in range(1, k):
                # Update distances
                current_point = xyz_b[indices[i-1]]
                dist_to_current = torch.norm(xyz_b - current_point, dim=1)
                distances = torch.min(distances, dist_to_current)
                
                # Select farthest point
                indices[i] = distances.argmax()
            
            sampled_xyz_list.append(xyz_b[indices])
            sampled_features_list.append(features_b[indices])
            indices_list.append(indices)
        
        # Stack batch
        sampled_xyz = torch.stack(sampled_xyz_list)
        sampled_features = torch.stack(sampled_features_list)
        indices = torch.stack(indices_list)
        
        return sampled_features, sampled_xyz, indices


class AblationStudy:
    """
    Comprehensive ablation study framework for analyzing sampling strategies.
    Essential for demonstrating the superiority of GAS.
    """
    
    def __init__(self, model: nn.Module, config):
        self.model = model
        self.config = config
        self.sampling_strategies = {
            'random': RandomSampling(),
            'grid': GridSampling(),
            'fps': FarthestPointSampling(),
            'gas': None  # Will use model's built-in GAS
        }
    
    def run_ablation(self, dataloader, device: torch.device) -> Dict[str, Dict[str, float]]:
        """
        Run ablation study comparing different sampling strategies.
        
        Returns:
            Dictionary mapping strategy names to their metrics
        """
        from boundary_metrics import BoundaryAwareMetrics, CurvatureAwareEvaluator
        
        results = {}
        
        for strategy_name, strategy in self.sampling_strategies.items():
            print(f"\nEvaluating {strategy_name} sampling...")
            
            # Create metrics computers
            boundary_metrics = BoundaryAwareMetrics(self.config.num_classes)
            curvature_evaluator = CurvatureAwareEvaluator(self.config.num_classes)
            
            # Temporarily replace model's sampling if needed
            if strategy_name != 'gas':
                original_gas_modules = self._replace_sampling(strategy)
            
            # Evaluate
            self.model.eval()
            with torch.no_grad():
                for batch in dataloader:
                    # Move to device
                    for key in batch:
                        if isinstance(batch[key], list):
                            for i in range(len(batch[key])):
                                batch[key][i] = batch[key][i].to(device)
                        else:
                            batch[key] = batch[key].to(device)
                    
                    # Forward pass
                    outputs = self.model(batch)
                    predictions = outputs['logits']
                    
                    # Update metrics
                    boundary_metrics.update(predictions, batch['labels'], batch['xyz'][0])
                    curvature_metrics = curvature_evaluator.evaluate(predictions, batch['labels'], batch['xyz'][0])
            
            # Restore original sampling
            if strategy_name != 'gas':
                self._restore_sampling(original_gas_modules)
            
            # Collect results
            results[strategy_name] = {
                **boundary_metrics.compute_metrics(),
                **curvature_metrics
            }
        
        return results
    
    def _replace_sampling(self, strategy: SamplingStrategy) -> List[nn.Module]:
        """Replace GAS modules with alternative sampling strategy."""
        original_modules = []
        
        class SamplingWrapper(nn.Module):
            def __init__(self, strategy, sample_ratio):
                super().__init__()
                self.strategy = strategy
                self.sample_ratio = sample_ratio
            
            def forward(self, xyz, features, sample_ratio):
                B, N, _ = features.shape
                k = max(min(int(np.floor(N / sample_ratio)), N), 1)
                sampled_features, sampled_xyz, indices = self.strategy.sample(xyz, features, k)
                # Return dummy aux_info to match GAS interface
                aux_info = {}
                return sampled_features, sampled_xyz, indices, aux_info
        
        # Replace in encoder
        for i, layer in enumerate(self.model.encoder_layers):
            original_modules.append(layer['gas'])
            layer['gas'] = SamplingWrapper(strategy, self.config.sub_sampling_ratio[i])
        
        return original_modules
    
    def _restore_sampling(self, original_modules: List[nn.Module]):
        """Restore original GAS modules."""
        for i, layer in enumerate(self.model.encoder_layers):
            layer['gas'] = original_modules[i]
    
    def analyze_results(self, results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Analyze ablation results to compute improvement metrics.
        
        Returns:
            Dictionary of improvement metrics
        """
        analysis = {}
        
        # Baseline is random sampling
        baseline = results['random']
        
        for strategy in results:
            if strategy == 'random':
                continue
            
            # Compute improvements
            strategy_results = results[strategy]
            
            # Overall improvement
            analysis[f'{strategy}_vs_random_mIoU'] = (
                (strategy_results['mIoU'] - baseline['mIoU']) / baseline['mIoU'] * 100
            )
            
            # Boundary improvement
            analysis[f'{strategy}_vs_random_boundary_mIoU'] = (
                (strategy_results['boundary_mIoU'] - baseline['boundary_mIoU']) / baseline['boundary_mIoU'] * 100
            )
            
            # High curvature improvement
            if 'mIoU_curvature_bin_4' in strategy_results:  # Highest curvature bin
                analysis[f'{strategy}_vs_random_high_curvature_mIoU'] = (
                    (strategy_results['mIoU_curvature_bin_4'] - baseline['mIoU_curvature_bin_4']) / baseline['mIoU_curvature_bin_4'] * 100
                )
        
        return analysis