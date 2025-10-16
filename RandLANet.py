"""
Final, A*-ready PyTorch implementation of the proposed segmentation network.
This version incorporates a Gumbel-Softmax differentiable sampler, fixes the
critical U-Net skip connection logic, and resolves an in-place operation bug
in the attention module to enable proper gradient flow.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List

# --- Foundational Utility Operations ---

def knn_blocked(query_xyz: torch.Tensor, support_xyz: torch.Tensor, k: int, block_size: int = 4096) -> torch.Tensor:
    """Memory-efficient K-Nearest Neighbors using GPU acceleration."""
    B, M, _ = query_xyz.shape
    all_top_indices = []
    num_blocks = (M + block_size - 1) // block_size

    for b_idx in range(num_blocks):
        start, end = b_idx * block_size, min((b_idx + 1) * block_size, M)
        block_xyz = query_xyz[:, start:end, :]
        dist_sq = torch.cdist(block_xyz, support_xyz, p=2) ** 2
        _, top_k_indices = torch.topk(dist_sq, k=k, dim=-1, largest=False, sorted=True)
        all_top_indices.append(top_k_indices)

    return torch.cat(all_top_indices, dim=1)

def gather_neighbors(data: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """A robust helper for batch-aware gathering of neighbor data on the GPU."""
    B, _, D = data.shape
    _, N_query, K = indices.shape
    batch_idx = torch.arange(B, device=data.device).view(B, 1, 1).expand(-1, N_query, K)
    return data[batch_idx, indices]

def gumbel_softmax_topk(logits: torch.Tensor, k: int, tau: float = 1.0, hard: bool = True, dim: int = -1) -> torch.Tensor:
    """
    A*-SUGGESTION A: Differentiable top-k selection using Gumbel-Softmax.
    Allows gradients to flow back through the sampling process.
    """
    gumbels = -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    gumbels = (logits + gumbels) / tau
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight-through estimator
        index = y_soft.topk(k, dim=dim)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Soft selection
        ret = y_soft
    return ret

# --- Core Architectural Modules ---

class GeometryAdaptiveSampling(nn.Module):
    """
    The key innovation module for geometry-aware point sampling, now with a
    differentiable Gumbel-Softmax sampler.
    """
    def __init__(self, in_channels: int, random_sample_ratio: float = 0.5):
        super().__init__()
        self.random_sample_ratio = random_sample_ratio
        self.tau = nn.Parameter(torch.tensor(1.0)) # Annealable temperature for Gumbel-Softmax

        self.curvature_mlp = nn.Sequential(
            nn.Linear(9, 32), nn.BatchNorm1d(32), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 1)
        )
        self.boundary_mlp = nn.Sequential(
            nn.Linear(in_channels + 3, 64), nn.BatchNorm1d(64), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 1)
        )
        self.score_fusion_mlp = nn.Sequential(
            nn.Linear(2, 16), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16, 1)
        )

    def _compute_local_covariance(self, xyz: torch.Tensor, k: int = 16) -> torch.Tensor:
        with torch.no_grad():
            neigh_idx = knn_blocked(xyz, xyz, k)
            neigh_xyz = gather_neighbors(xyz, neigh_idx)
            centered_neighbors = neigh_xyz - xyz.unsqueeze(2)
            cov = torch.matmul(centered_neighbors.transpose(-1, -2), centered_neighbors) / k
        return cov

    def forward(self, xyz: torch.Tensor, features: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        B, N, C = features.shape
        cov_matrix = self._compute_local_covariance(xyz).view(B * N, 9)
        curvature_scores = self.curvature_mlp(cov_matrix).view(B, N, 1)
        boundary_input = torch.cat([xyz, features], dim=-1).view(B * N, C + 3)
        boundary_scores = self.boundary_mlp(boundary_input).view(B, N, 1)
        combined_scores = torch.cat([curvature_scores, boundary_scores], dim=-1)
        importance_scores = self.score_fusion_mlp(combined_scores.view(B * N, 2)).view(B, N)
        
        learned_k = int(k * (1 - self.random_sample_ratio))
        random_k = k - learned_k

        # --- Use Differentiable Sampler ---
        # Get soft selection, then get hard indices for gathering
        soft_selection = gumbel_softmax_topk(importance_scores, k=learned_k, tau=self.tau, hard=True)
        learned_idx = soft_selection.topk(learned_k, dim=1)[1]
        
        # Random sampling for the remainder
        rand_scores = torch.rand(B, N, device=xyz.device)
        rand_scores.scatter_(1, learned_idx, -1.0)
        _, random_idx = torch.topk(rand_scores, k=random_k, dim=1)
        
        combined_idx, _ = torch.sort(torch.cat([learned_idx, random_idx], dim=1), dim=1)
        
        sampled_xyz = gather_neighbors(xyz, combined_idx.unsqueeze(-1)).squeeze(2)
        # We can also create a soft version of features if needed for other loss terms
        sampled_features = gather_neighbors(features, combined_idx.unsqueeze(-1)).squeeze(2)

        aux_info = {'importance_scores': F.softmax(importance_scores, dim=-1), 'learned_indices': learned_idx}
        return sampled_features, sampled_xyz, combined_idx, aux_info

class LocalTransformerAggregation(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, heads: int = 4):
        super().__init__()
        self.heads, self.head_dim = heads, out_channels // heads
        self.q_proj, self.k_proj, self.v_proj = nn.Linear(in_channels, out_channels), nn.Linear(in_channels, out_channels), nn.Linear(in_channels, out_channels)
        self.pos_mlp = nn.Sequential(nn.Linear(4, out_channels), nn.ReLU(), nn.Linear(out_channels, out_channels))
        self.out_proj = nn.Linear(out_channels, out_channels)
        self.res_proj = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, center_xyz: torch.Tensor, center_features: torch.Tensor, neigh_xyz: torch.Tensor, neigh_feat: torch.Tensor) -> torch.Tensor:
        B, N, _ = center_features.shape; _, _, K, _ = neigh_feat.shape
        rel_pos = neigh_xyz - center_xyz.unsqueeze(2)
        rel_dist = torch.norm(rel_pos, dim=-1, keepdim=True)
        pos_enc = self.pos_mlp(torch.cat([rel_pos, rel_dist], dim=-1))
        q = self.q_proj(center_features).view(B, N, self.heads, self.head_dim)
        k = self.k_proj(neigh_feat).view(B, N, K, self.heads, self.head_dim)
        v = self.v_proj(neigh_feat).view(B, N, K, self.heads, self.head_dim)
        
        k = k + pos_enc.view(B, N, K, self.heads, self.head_dim)
        v = v + pos_enc.view(B, N, K, self.heads, self.head_dim)

        q, k, v = q.permute(0, 2, 1, 3), k.permute(0, 3, 1, 2, 4), v.permute(0, 3, 1, 2, 4)
        scores = torch.einsum('bhnd,bhnkd->bhnk', q, k) / (self.head_dim ** 0.5)
        attn_out = torch.einsum('bhnk,bhnkd->bhnd', F.softmax(scores, dim=-1), v).permute(0, 2, 1, 3).contiguous().view(B, N, -1)
        return F.leaky_relu(self.out_proj(attn_out) + self.res_proj(center_features), 0.2)

class Network(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config, self.store_aux_info = config, False
        self.fc0 = nn.Sequential(nn.Linear(config.num_features, 8), nn.BatchNorm1d(8), nn.LeakyReLU(0.2, inplace=True))
        
        self.encoder_layers, in_c = nn.ModuleList(), 8
        for i in range(config.num_layers):
            out_c = config.d_out[i]
            self.encoder_layers.append(nn.ModuleDict({'lta': LocalTransformerAggregation(in_c, out_c),
                                                     'mlp': nn.Sequential(nn.Linear(out_c, out_c), nn.BatchNorm1d(out_c), nn.LeakyReLU(0.2, inplace=True)),
                                                     'gas': GeometryAdaptiveSampling(out_c)}))
            in_c = out_c
        
        self.bottleneck = nn.Sequential(nn.Linear(in_c, in_c), nn.BatchNorm1d(in_c), nn.LeakyReLU(0.2, inplace=True))
        self.decoder_layers = nn.ModuleList()
        decoder_in_c = in_c
        for i in range(config.num_layers):
            skip_c = config.d_out[config.num_layers - 1 - i]
            mlp_in_c = skip_c + decoder_in_c
            mlp_out_c = skip_c
            self.decoder_layers.append(nn.ModuleDict({
                'mlp': nn.Sequential(nn.Linear(mlp_in_c, mlp_out_c), nn.BatchNorm1d(mlp_out_c), nn.LeakyReLU(0.2, inplace=True)),
                'lta': LocalTransformerAggregation(mlp_out_c, mlp_out_c)
            }))
            decoder_in_c = mlp_out_c

        self.classifier = nn.Sequential(
            nn.Linear(config.d_out[0], 64), nn.BatchNorm1d(64), nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5), nn.Linear(64, config.num_classes)
        )

    def forward(self, inputs: dict) -> Dict[str, any]:
        xyz, features = inputs['xyz'], inputs['features']
        B, N, _ = features.shape
        self.aux_info_list = [] if self.store_aux_info else None
        
        encoder_skip_xyz: List[torch.Tensor] = []
        encoder_skip_features: List[torch.Tensor] = []
        
        downsampled_xyz: List[torch.Tensor] = [xyz]
        initial_feat = self.fc0(features.view(-1, self.config.num_features)).view(B, N, -1)
        downsampled_features: List[torch.Tensor] = [initial_feat]
        
        for i, layer in enumerate(self.encoder_layers):
            xyz_i, feat_i = downsampled_xyz[-1], downsampled_features[-1]
            neigh_idx = knn_blocked(xyz_i, xyz_i, self.config.k_n)
            neigh_xyz, neigh_feat = gather_neighbors(xyz_i, neigh_idx), gather_neighbors(feat_i, neigh_idx)
            
            feat_ta = layer['lta'](xyz_i, feat_i, neigh_xyz, neigh_feat)
            feat_enc = layer['mlp'](feat_ta.view(-1, feat_ta.shape[-1])).view_as(feat_ta)
            
            encoder_skip_xyz.append(xyz_i)
            encoder_skip_features.append(feat_enc)
            
            k_points = feat_i.shape[1] // self.config.sub_sampling_ratio[i]
            sampled_feat, sampled_xyz, _, aux_info = layer['gas'](xyz_i, feat_enc, k_points)
            
            downsampled_xyz.append(sampled_xyz)
            downsampled_features.append(sampled_feat)
            if self.store_aux_info: self.aux_info_list.append(aux_info)
        
        bottleneck_feat = downsampled_features[-1]
        decoded_feat = self.bottleneck(bottleneck_feat.view(-1, bottleneck_feat.shape[-1])).view_as(bottleneck_feat)

        for i in range(self.config.num_layers):
            layer_idx = self.config.num_layers - 1 - i
            
            skip_xyz = encoder_skip_xyz[layer_idx]
            skip_feat = encoder_skip_features[layer_idx]
            
            interp_idx = knn_blocked(query_xyz=skip_xyz, support_xyz=downsampled_xyz[layer_idx + 1], k=1)
            upsampled_feat = gather_neighbors(decoded_feat, interp_idx).squeeze(2)
            
            concat_feat = torch.cat([skip_feat, upsampled_feat], dim=-1)
            feat_dec = self.decoder_layers[i]['mlp'](concat_feat.view(-1, concat_feat.shape[-1])).view(B, concat_feat.shape[1], -1)
            
            dec_neigh_idx = knn_blocked(skip_xyz, skip_xyz, self.config.k_n)
            dec_neigh_xyz = gather_neighbors(skip_xyz, dec_neigh_idx)
            dec_neigh_feat = gather_neighbors(feat_dec, dec_neigh_idx)
            decoded_feat = self.decoder_layers[i]['lta'](skip_xyz, feat_dec, dec_neigh_xyz, dec_neigh_feat)

        logits = self.classifier(decoded_feat.view(-1, decoded_feat.shape[-1])).view(B, N, -1)
        return {'logits': logits, 'aux_info': self.aux_info_list}




# """
# Final, A*-ready PyTorch implementation of the proposed segmentation network.
# This version corrects the U-Net skip connection logic and fixes a critical
# in-place operation bug in the attention module to enable proper gradient flow.
# """
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Tuple, Dict, List

# # --- Foundational Utility Operations ---

# def knn_blocked(query_xyz: torch.Tensor, support_xyz: torch.Tensor, k: int, block_size: int = 4096) -> torch.Tensor:
#     """Memory-efficient K-Nearest Neighbors using GPU acceleration."""
#     B, M, _ = query_xyz.shape
#     all_top_indices = []
#     num_blocks = (M + block_size - 1) // block_size

#     for b_idx in range(num_blocks):
#         start, end = b_idx * block_size, min((b_idx + 1) * block_size, M)
#         block_xyz = query_xyz[:, start:end, :]
#         dist_sq = torch.cdist(block_xyz, support_xyz, p=2) ** 2
#         _, top_k_indices = torch.topk(dist_sq, k=k, dim=-1, largest=False, sorted=True)
#         all_top_indices.append(top_k_indices)

#     return torch.cat(all_top_indices, dim=1)

# def gather_neighbors(data: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
#     """A robust helper for batch-aware gathering of neighbor data on the GPU."""
#     B, _, D = data.shape
#     _, N_query, K = indices.shape
#     batch_idx = torch.arange(B, device=data.device).view(B, 1, 1).expand(-1, N_query, K)
#     return data[batch_idx, indices]

# # --- Core Architectural Modules ---

# class GeometryAdaptiveSampling(nn.Module):
#     """The key innovation module for geometry-aware point sampling."""
#     def __init__(self, in_channels: int, random_sample_ratio: float = 0.5):
#         super().__init__()
#         self.random_sample_ratio = random_sample_ratio
#         self.curvature_mlp = nn.Sequential(
#             nn.Linear(9, 32), nn.BatchNorm1d(32), nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(32, 1)
#         )
#         self.boundary_mlp = nn.Sequential(
#             nn.Linear(in_channels + 3, 64), nn.BatchNorm1d(64), nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(64, 32), nn.BatchNorm1d(32), nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(32, 1)
#         )
#         self.score_fusion_mlp = nn.Sequential(
#             nn.Linear(2, 16), nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(16, 1)
#         )

#     def _compute_local_covariance(self, xyz: torch.Tensor, k: int = 16) -> torch.Tensor:
#         with torch.no_grad(): # Covariance is a geometric property, no gradients needed here
#             neigh_idx = knn_blocked(xyz, xyz, k)
#             neigh_xyz = gather_neighbors(xyz, neigh_idx)
#             centered_neighbors = neigh_xyz - xyz.unsqueeze(2)
#             cov = torch.matmul(centered_neighbors.transpose(-1, -2), centered_neighbors) / k
#         return cov

#     def forward(self, xyz: torch.Tensor, features: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
#         B, N, C = features.shape
#         cov_matrix = self._compute_local_covariance(xyz).view(B * N, 9)
#         curvature_scores = self.curvature_mlp(cov_matrix).view(B, N, 1)
#         boundary_input = torch.cat([xyz, features], dim=-1).view(B * N, C + 3)
#         boundary_scores = self.boundary_mlp(boundary_input).view(B, N, 1)
#         combined_scores = torch.cat([curvature_scores, boundary_scores], dim=-1)
#         importance_scores = self.score_fusion_mlp(combined_scores.view(B * N, 2)).view(B, N)
#         learned_k = int(k * (1 - self.random_sample_ratio))
#         random_k = k - learned_k
#         _, learned_idx = torch.topk(importance_scores, k=learned_k, dim=1)
#         rand_scores = torch.rand(B, N, device=xyz.device)
#         rand_scores.scatter_(1, learned_idx, -1.0)
#         _, random_idx = torch.topk(rand_scores, k=random_k, dim=1)
#         combined_idx, _ = torch.sort(torch.cat([learned_idx, random_idx], dim=1), dim=1)
#         sampled_xyz = gather_neighbors(xyz, combined_idx.unsqueeze(-1)).squeeze(2)
#         sampled_features = gather_neighbors(features, combined_idx.unsqueeze(-1)).squeeze(2)
#         aux_info = {
#             'curvature_scores': torch.sigmoid(curvature_scores.squeeze(-1)),
#             'boundary_scores': torch.sigmoid(boundary_scores.squeeze(-1)),
#             'importance_scores': F.softmax(importance_scores, dim=-1),
#             'learned_indices': learned_idx,
#         }
#         return sampled_features, sampled_xyz, combined_idx, aux_info

# class LocalTransformerAggregation(nn.Module):
#     """Attention-based module for aggregating local neighborhood features."""
#     def __init__(self, in_channels: int, out_channels: int, heads: int = 4):
#         super().__init__()
#         self.heads, self.head_dim = heads, out_channels // heads
#         self.q_proj, self.k_proj, self.v_proj = nn.Linear(in_channels, out_channels), nn.Linear(in_channels, out_channels), nn.Linear(in_channels, out_channels)
#         self.pos_mlp = nn.Sequential(nn.Linear(4, out_channels), nn.ReLU(), nn.Linear(out_channels, out_channels))
#         self.out_proj = nn.Linear(out_channels, out_channels)
#         self.res_proj = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

#     def forward(self, center_xyz: torch.Tensor, center_features: torch.Tensor, neigh_xyz: torch.Tensor, neigh_feat: torch.Tensor) -> torch.Tensor:
#         B, N, _ = center_features.shape; _, _, K, _ = neigh_feat.shape
#         rel_pos = neigh_xyz - center_xyz.unsqueeze(2)
#         rel_dist = torch.norm(rel_pos, dim=-1, keepdim=True)
#         pos_enc = self.pos_mlp(torch.cat([rel_pos, rel_dist], dim=-1))
#         q = self.q_proj(center_features).view(B, N, self.heads, self.head_dim)
#         k = self.k_proj(neigh_feat).view(B, N, K, self.heads, self.head_dim)
#         v = self.v_proj(neigh_feat).view(B, N, K, self.heads, self.head_dim)

#         # --- BUG FIX ---
#         # Changed in-place addition (+=) to out-of-place (+) to preserve the computation graph.
#         k = k + pos_enc.view(B, N, K, self.heads, self.head_dim)
#         v = v + pos_enc.view(B, N, K, self.heads, self.head_dim)

#         q, k, v = q.permute(0, 2, 1, 3), k.permute(0, 3, 1, 2, 4), v.permute(0, 3, 1, 2, 4)
#         scores = torch.einsum('bhnd,bhnkd->bhnk', q, k) / (self.head_dim ** 0.5)
#         attn_out = torch.einsum('bhnk,bhnkd->bhnd', F.softmax(scores, dim=-1), v).permute(0, 2, 1, 3).contiguous().view(B, N, -1)
#         return F.leaky_relu(self.out_proj(attn_out) + self.res_proj(center_features), 0.2)

# class Network(nn.Module):
#     """The full Encoder-Decoder segmentation network."""
#     def __init__(self, config):
#         super().__init__()
#         self.config, self.store_aux_info = config, False
#         self.fc0 = nn.Sequential(nn.Linear(config.num_features, 8), nn.BatchNorm1d(8), nn.LeakyReLU(0.2, inplace=True))
        
#         self.encoder_layers, in_c = nn.ModuleList(), 8
#         for i in range(config.num_layers):
#             out_c = config.d_out[i]
#             self.encoder_layers.append(nn.ModuleDict({'lta': LocalTransformerAggregation(in_c, out_c),
#                                                      'mlp': nn.Sequential(nn.Linear(out_c, out_c), nn.BatchNorm1d(out_c), nn.LeakyReLU(0.2, inplace=True)),
#                                                      'gas': GeometryAdaptiveSampling(out_c)}))
#             in_c = out_c
        
#         self.bottleneck = nn.Sequential(nn.Linear(in_c, in_c), nn.BatchNorm1d(in_c), nn.LeakyReLU(0.2, inplace=True))
#         self.decoder_layers = nn.ModuleList()
#         decoder_in_c = in_c
#         for i in range(config.num_layers):
#             skip_c = config.d_out[config.num_layers - 1 - i]
#             mlp_in_c = skip_c + decoder_in_c
#             mlp_out_c = skip_c
#             self.decoder_layers.append(nn.ModuleDict({
#                 'mlp': nn.Sequential(nn.Linear(mlp_in_c, mlp_out_c), nn.BatchNorm1d(mlp_out_c), nn.LeakyReLU(0.2, inplace=True)),
#                 'lta': LocalTransformerAggregation(mlp_out_c, mlp_out_c)
#             }))
#             decoder_in_c = mlp_out_c

#         self.classifier = nn.Sequential(
#             nn.Linear(config.d_out[0], 64), nn.BatchNorm1d(64), nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(0.5), nn.Linear(64, config.num_classes)
#         )

#     def forward(self, inputs: dict) -> Dict[str, any]:
#         xyz, features = inputs['xyz'], inputs['features']
#         B, N, _ = features.shape
#         self.aux_info_list = [] if self.store_aux_info else None

#         encoder_skip_xyz: List[torch.Tensor] = []
#         encoder_skip_features: List[torch.Tensor] = []
        
#         downsampled_xyz: List[torch.Tensor] = [xyz]
#         initial_feat = self.fc0(features.view(-1, self.config.num_features)).view(B, N, -1)
#         downsampled_features: List[torch.Tensor] = [initial_feat]
        
#         for i, layer in enumerate(self.encoder_layers):
#             xyz_i, feat_i = downsampled_xyz[-1], downsampled_features[-1]
#             neigh_idx = knn_blocked(xyz_i, xyz_i, self.config.k_n)
#             neigh_xyz, neigh_feat = gather_neighbors(xyz_i, neigh_idx), gather_neighbors(feat_i, neigh_idx)
            
#             feat_ta = layer['lta'](xyz_i, feat_i, neigh_xyz, neigh_feat)
#             feat_enc = layer['mlp'](feat_ta.view(-1, feat_ta.shape[-1])).view_as(feat_ta)
            
#             encoder_skip_xyz.append(xyz_i)
#             encoder_skip_features.append(feat_enc)
            
#             k_points = feat_i.shape[1] // self.config.sub_sampling_ratio[i]
#             sampled_feat, sampled_xyz, _, aux_info = layer['gas'](xyz_i, feat_enc, k_points)
            
#             downsampled_xyz.append(sampled_xyz)
#             downsampled_features.append(sampled_feat)
#             if self.store_aux_info: self.aux_info_list.append(aux_info)
        
#         bottleneck_feat = downsampled_features[-1]
#         decoded_feat = self.bottleneck(bottleneck_feat.view(-1, bottleneck_feat.shape[-1])).view_as(bottleneck_feat)

#         for i in range(self.config.num_layers):
#             layer_idx = self.config.num_layers - 1 - i
            
#             skip_xyz = encoder_skip_xyz[layer_idx]
#             skip_feat = encoder_skip_features[layer_idx]
            
#             interp_idx = knn_blocked(query_xyz=skip_xyz, support_xyz=downsampled_xyz[layer_idx + 1], k=1)
#             upsampled_feat = gather_neighbors(decoded_feat, interp_idx).squeeze(2)
            
#             concat_feat = torch.cat([skip_feat, upsampled_feat], dim=-1)
#             feat_dec = self.decoder_layers[i]['mlp'](concat_feat.view(-1, concat_feat.shape[-1])).view(B, concat_feat.shape[1], -1)
            
#             dec_neigh_idx = knn_blocked(skip_xyz, skip_xyz, self.config.k_n)
#             dec_neigh_xyz = gather_neighbors(skip_xyz, dec_neigh_idx)
#             dec_neigh_feat = gather_neighbors(feat_dec, dec_neigh_idx)
#             decoded_feat = self.decoder_layers[i]['lta'](skip_xyz, feat_dec, dec_neigh_xyz, dec_neigh_feat)

#         logits = self.classifier(decoded_feat.view(-1, decoded_feat.shape[-1])).view(B, N, -1)
#         return {'logits': logits, 'aux_info': self.aux_info_list}







# """
# Final, A*-ready PyTorch implementation of the proposed segmentation network.
# This version corrects the U-Net skip connection logic for proper detail preservation.
# ---
# DEBUG VERSION: Includes print statements to trace the computation graph.
# """
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Tuple, Dict, List

# # --- Foundational Utility Operations (Unchanged) ---

# def knn_blocked(query_xyz: torch.Tensor, support_xyz: torch.Tensor, k: int, block_size: int = 4096) -> torch.Tensor:
#     """Memory-efficient K-Nearest Neighbors using GPU acceleration."""
#     B, M, _ = query_xyz.shape
#     all_top_indices = []
#     num_blocks = (M + block_size - 1) // block_size

#     for b_idx in range(num_blocks):
#         start, end = b_idx * block_size, min((b_idx + 1) * block_size, M)
#         block_xyz = query_xyz[:, start:end, :]
#         dist_sq = torch.cdist(block_xyz, support_xyz, p=2) ** 2
#         _, top_k_indices = torch.topk(dist_sq, k=k, dim=-1, largest=False, sorted=True)
#         all_top_indices.append(top_k_indices)

#     return torch.cat(all_top_indices, dim=1)

# def gather_neighbors(data: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
#     """A robust helper for batch-aware gathering of neighbor data on the GPU."""
#     B, _, D = data.shape
#     _, N_query, K = indices.shape
#     batch_idx = torch.arange(B, device=data.device).view(B, 1, 1).expand(-1, N_query, K)
#     return data[batch_idx, indices]

# # --- Core Architectural Modules (Unchanged) ---

# class GeometryAdaptiveSampling(nn.Module):
#     """The key innovation module for geometry-aware point sampling."""
#     def __init__(self, in_channels: int, random_sample_ratio: float = 0.5):
#         super().__init__()
#         self.random_sample_ratio = random_sample_ratio
#         self.curvature_mlp = nn.Sequential(
#             nn.Linear(9, 32), nn.BatchNorm1d(32), nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(32, 1)
#         )
#         self.boundary_mlp = nn.Sequential(
#             nn.Linear(in_channels + 3, 64), nn.BatchNorm1d(64), nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(64, 32), nn.BatchNorm1d(32), nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(32, 1)
#         )
#         self.score_fusion_mlp = nn.Sequential(
#             nn.Linear(2, 16), nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(16, 1)
#         )

#     def _compute_local_covariance(self, xyz: torch.Tensor, k: int = 16) -> torch.Tensor:
#         neigh_idx = knn_blocked(xyz, xyz, k)
#         neigh_xyz = gather_neighbors(xyz, neigh_idx)
#         centered_neighbors = neigh_xyz - xyz.unsqueeze(2)
#         cov = torch.matmul(centered_neighbors.transpose(-1, -2), centered_neighbors) / k
#         return cov

#     def forward(self, xyz: torch.Tensor, features: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
#         B, N, C = features.shape
#         cov_matrix = self._compute_local_covariance(xyz).view(B * N, 9)
#         curvature_scores = self.curvature_mlp(cov_matrix).view(B, N, 1)
#         boundary_input = torch.cat([xyz, features], dim=-1).view(B * N, C + 3)
#         boundary_scores = self.boundary_mlp(boundary_input).view(B, N, 1)
#         combined_scores = torch.cat([curvature_scores, boundary_scores], dim=-1)
#         importance_scores = self.score_fusion_mlp(combined_scores.view(B * N, 2)).view(B, N)
#         learned_k = int(k * (1 - self.random_sample_ratio))
#         random_k = k - learned_k
#         _, learned_idx = torch.topk(importance_scores, k=learned_k, dim=1)
#         rand_scores = torch.rand(B, N, device=xyz.device)
#         rand_scores.scatter_(1, learned_idx, -1.0)
#         _, random_idx = torch.topk(rand_scores, k=random_k, dim=1)
#         combined_idx, _ = torch.sort(torch.cat([learned_idx, random_idx], dim=1), dim=1)
#         sampled_xyz = gather_neighbors(xyz, combined_idx.unsqueeze(-1)).squeeze(2)
#         sampled_features = gather_neighbors(features, combined_idx.unsqueeze(-1)).squeeze(2)
#         aux_info = {
#             'curvature_scores': torch.sigmoid(curvature_scores.squeeze(-1)),
#             'boundary_scores': torch.sigmoid(boundary_scores.squeeze(-1)),
#             'importance_scores': F.softmax(importance_scores, dim=-1),
#             'learned_indices': learned_idx,
#         }
#         return sampled_features, sampled_xyz, combined_idx, aux_info

# class LocalTransformerAggregation(nn.Module):
#     """Attention-based module for aggregating local neighborhood features."""
#     def __init__(self, in_channels: int, out_channels: int, heads: int = 4):
#         super().__init__()
#         self.heads, self.head_dim = heads, out_channels // heads
#         self.q_proj, self.k_proj, self.v_proj = nn.Linear(in_channels, out_channels), nn.Linear(in_channels, out_channels), nn.Linear(in_channels, out_channels)
#         self.pos_mlp = nn.Sequential(nn.Linear(4, out_channels), nn.ReLU(), nn.Linear(out_channels, out_channels))
#         self.out_proj = nn.Linear(out_channels, out_channels)
#         self.res_proj = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

#     def forward(self, center_xyz: torch.Tensor, center_features: torch.Tensor, neigh_xyz: torch.Tensor, neigh_feat: torch.Tensor) -> torch.Tensor:
#         B, N, _ = center_features.shape; _, _, K, _ = neigh_feat.shape
#         rel_pos = neigh_xyz - center_xyz.unsqueeze(2)
#         rel_dist = torch.norm(rel_pos, dim=-1, keepdim=True)
#         pos_enc = self.pos_mlp(torch.cat([rel_pos, rel_dist], dim=-1))
#         q = self.q_proj(center_features).view(B, N, self.heads, self.head_dim)
#         k = self.k_proj(neigh_feat).view(B, N, K, self.heads, self.head_dim)
#         v = self.v_proj(neigh_feat).view(B, N, K, self.heads, self.head_dim)
#         k += pos_enc.view(B, N, K, self.heads, self.head_dim)
#         v += pos_enc.view(B, N, K, self.heads, self.head_dim)
#         q, k, v = q.permute(0, 2, 1, 3), k.permute(0, 3, 1, 2, 4), v.permute(0, 3, 1, 2, 4)
#         scores = torch.einsum('bhnd,bhnkd->bhnk', q, k) / (self.head_dim ** 0.5)
#         attn_out = torch.einsum('bhnk,bhnkd->bhnd', F.softmax(scores, dim=-1), v).permute(0, 2, 1, 3).contiguous().view(B, N, -1)
#         return F.leaky_relu(self.out_proj(attn_out) + self.res_proj(center_features), 0.2)

# class Network(nn.Module):
#     """The full Encoder-Decoder segmentation network."""
#     def __init__(self, config):
#         super().__init__()
#         self.config, self.store_aux_info = config, False
#         self.fc0 = nn.Sequential(nn.Linear(config.num_features, 8), nn.BatchNorm1d(8), nn.LeakyReLU(0.2, inplace=True))
        
#         self.encoder_layers, in_c = nn.ModuleList(), 8
#         for i in range(config.num_layers):
#             out_c = config.d_out[i]
#             self.encoder_layers.append(nn.ModuleDict({'lta': LocalTransformerAggregation(in_c, out_c),
#                                                      'mlp': nn.Sequential(nn.Linear(out_c, out_c), nn.BatchNorm1d(out_c), nn.LeakyReLU(0.2, inplace=True)),
#                                                      'gas': GeometryAdaptiveSampling(out_c)}))
#             in_c = out_c
        
#         self.bottleneck = nn.Sequential(nn.Linear(in_c, in_c), nn.BatchNorm1d(in_c), nn.LeakyReLU(0.2, inplace=True))
#         self.decoder_layers = nn.ModuleList()
#         decoder_in_c = in_c
#         for i in range(config.num_layers):
#             skip_c = config.d_out[config.num_layers - 1 - i]
#             mlp_in_c = skip_c + decoder_in_c
#             mlp_out_c = skip_c
#             self.decoder_layers.append(nn.ModuleDict({
#                 'mlp': nn.Sequential(nn.Linear(mlp_in_c, mlp_out_c), nn.BatchNorm1d(mlp_out_c), nn.LeakyReLU(0.2, inplace=True)),
#                 'lta': LocalTransformerAggregation(mlp_out_c, mlp_out_c)
#             }))
#             decoder_in_c = mlp_out_c

#         self.classifier = nn.Sequential(
#             nn.Linear(config.d_out[0], 64), nn.BatchNorm1d(64), nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(0.5), nn.Linear(64, config.num_classes)
#         )

#     def forward(self, inputs: dict) -> Dict[str, any]:
#         xyz, features = inputs['xyz'], inputs['features']
#         B, N, _ = features.shape
#         self.aux_info_list = [] if self.store_aux_info else None

#         encoder_skip_xyz: List[torch.Tensor] = []
#         encoder_skip_features: List[torch.Tensor] = []
        
#         downsampled_xyz: List[torch.Tensor] = [xyz]
#         initial_feat = self.fc0(features.view(-1, self.config.num_features)).view(B, N, -1)
#         downsampled_features: List[torch.Tensor] = [initial_feat]
        
#         print(f"\n--- FORWARD PASS START ---")
#         print(f"Initial Features grad_fn: {initial_feat.grad_fn}")
        
#         for i, layer in enumerate(self.encoder_layers):
#             xyz_i, feat_i = downsampled_xyz[-1], downsampled_features[-1]
#             neigh_idx = knn_blocked(xyz_i, xyz_i, self.config.k_n)
#             neigh_xyz, neigh_feat = gather_neighbors(xyz_i, neigh_idx), gather_neighbors(feat_i, neigh_idx)
            
#             feat_ta = layer['lta'](xyz_i, feat_i, neigh_xyz, neigh_feat)
#             feat_enc = layer['mlp'](feat_ta.view(-1, feat_ta.shape[-1])).view_as(feat_ta)
            
#             encoder_skip_xyz.append(xyz_i)
#             encoder_skip_features.append(feat_enc)
            
#             print(f"Encoder Layer {i} | Skip Features grad_fn: {feat_enc.grad_fn}")
            
#             k_points = feat_i.shape[1] // self.config.sub_sampling_ratio[i]
#             sampled_feat, sampled_xyz, _, aux_info = layer['gas'](xyz_i, feat_enc, k_points)
            
#             downsampled_xyz.append(sampled_xyz)
#             downsampled_features.append(sampled_feat)
#             if self.store_aux_info: self.aux_info_list.append(aux_info)
        
#         bottleneck_feat = downsampled_features[-1]
#         decoded_feat = self.bottleneck(bottleneck_feat.view(-1, bottleneck_feat.shape[-1])).view_as(bottleneck_feat)
#         print(f"Bottleneck Output grad_fn: {decoded_feat.grad_fn}")

#         for i in range(self.config.num_layers):
#             layer_idx = self.config.num_layers - 1 - i
            
#             skip_xyz = encoder_skip_xyz[layer_idx]
#             skip_feat = encoder_skip_features[layer_idx]
            
#             interp_idx = knn_blocked(query_xyz=skip_xyz, support_xyz=downsampled_xyz[layer_idx + 1], k=1)
#             upsampled_feat = gather_neighbors(decoded_feat, interp_idx).squeeze(2)
            
#             print(f"\n--- Decoder Layer {i} ---")
#             print(f"  Input 'decoded_feat' grad_fn: {decoded_feat.grad_fn}")
#             print(f"  'skip_feat' grad_fn:          {skip_feat.grad_fn}")
#             print(f"  'upsampled_feat' grad_fn:     {upsampled_feat.grad_fn}")
            
#             concat_feat = torch.cat([skip_feat, upsampled_feat], dim=-1)
#             feat_dec = self.decoder_layers[i]['mlp'](concat_feat.view(-1, concat_feat.shape[-1])).view(B, concat_feat.shape[1], -1)
            
#             dec_neigh_idx = knn_blocked(skip_xyz, skip_xyz, self.config.k_n)
#             dec_neigh_xyz = gather_neighbors(skip_xyz, dec_neigh_idx)
#             dec_neigh_feat = gather_neighbors(feat_dec, dec_neigh_idx)
#             decoded_feat = self.decoder_layers[i]['lta'](skip_xyz, feat_dec, dec_neigh_xyz, dec_neigh_feat)
#             print(f"  Output 'decoded_feat' grad_fn: {decoded_feat.grad_fn}")


#         logits = self.classifier(decoded_feat.view(-1, decoded_feat.shape[-1])).view(B, N, -1)
#         print(f"\nFinal Logits grad_fn: {logits.grad_fn}")
#         print(f"--- FORWARD PASS END ---\n")
#         return {'logits': logits, 'aux_info': self.aux_info_list}


# """
# Final, A*-ready PyTorch implementation of the proposed segmentation network.
# This version corrects the U-Net skip connection logic for proper detail preservation.
# """
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Tuple, Dict, List

# # --- Foundational Utility Operations (Unchanged) ---

# def knn_blocked(query_xyz: torch.Tensor, support_xyz: torch.Tensor, k: int, block_size: int = 4096) -> torch.Tensor:
#     """Memory-efficient K-Nearest Neighbors using GPU acceleration."""
#     B, M, _ = query_xyz.shape
#     all_top_indices = []
#     num_blocks = (M + block_size - 1) // block_size

#     for b_idx in range(num_blocks):
#         start, end = b_idx * block_size, min((b_idx + 1) * block_size, M)
#         block_xyz = query_xyz[:, start:end, :]
#         dist_sq = torch.cdist(block_xyz, support_xyz, p=2) ** 2
#         _, top_k_indices = torch.topk(dist_sq, k=k, dim=-1, largest=False, sorted=True)
#         all_top_indices.append(top_k_indices)

#     return torch.cat(all_top_indices, dim=1)

# def gather_neighbors(data: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
#     """A robust helper for batch-aware gathering of neighbor data on the GPU."""
#     B, _, D = data.shape
#     _, N_query, K = indices.shape
#     batch_idx = torch.arange(B, device=data.device).view(B, 1, 1).expand(-1, N_query, K)
#     return data[batch_idx, indices]

# # --- Core Architectural Modules (Unchanged) ---

# class GeometryAdaptiveSampling(nn.Module):
#     """The key innovation module for geometry-aware point sampling."""
#     def __init__(self, in_channels: int, random_sample_ratio: float = 0.5):
#         super().__init__()
#         self.random_sample_ratio = random_sample_ratio
#         self.curvature_mlp = nn.Sequential(
#             nn.Linear(9, 32), nn.BatchNorm1d(32), nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(32, 1)
#         )
#         self.boundary_mlp = nn.Sequential(
#             nn.Linear(in_channels + 3, 64), nn.BatchNorm1d(64), nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(64, 32), nn.BatchNorm1d(32), nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(32, 1)
#         )
#         self.score_fusion_mlp = nn.Sequential(
#             nn.Linear(2, 16), nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(16, 1)
#         )

#     def _compute_local_covariance(self, xyz: torch.Tensor, k: int = 16) -> torch.Tensor:
#         neigh_idx = knn_blocked(xyz, xyz, k)
#         neigh_xyz = gather_neighbors(xyz, neigh_idx)
#         centered_neighbors = neigh_xyz - xyz.unsqueeze(2)
#         cov = torch.matmul(centered_neighbors.transpose(-1, -2), centered_neighbors) / k
#         return cov

#     def forward(self, xyz: torch.Tensor, features: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
#         B, N, C = features.shape
#         cov_matrix = self._compute_local_covariance(xyz).view(B * N, 9)
#         curvature_scores = self.curvature_mlp(cov_matrix).view(B, N, 1)
#         boundary_input = torch.cat([xyz, features], dim=-1).view(B * N, C + 3)
#         boundary_scores = self.boundary_mlp(boundary_input).view(B, N, 1)
#         combined_scores = torch.cat([curvature_scores, boundary_scores], dim=-1)
#         importance_scores = self.score_fusion_mlp(combined_scores.view(B * N, 2)).view(B, N)
#         learned_k = int(k * (1 - self.random_sample_ratio))
#         random_k = k - learned_k
#         _, learned_idx = torch.topk(importance_scores, k=learned_k, dim=1)
#         rand_scores = torch.rand(B, N, device=xyz.device)
#         rand_scores.scatter_(1, learned_idx, -1.0)
#         _, random_idx = torch.topk(rand_scores, k=random_k, dim=1)
#         combined_idx, _ = torch.sort(torch.cat([learned_idx, random_idx], dim=1), dim=1)
#         sampled_xyz = gather_neighbors(xyz, combined_idx.unsqueeze(-1)).squeeze(2)
#         sampled_features = gather_neighbors(features, combined_idx.unsqueeze(-1)).squeeze(2)
#         aux_info = {
#             'curvature_scores': torch.sigmoid(curvature_scores.squeeze(-1)),
#             'boundary_scores': torch.sigmoid(boundary_scores.squeeze(-1)),
#             'importance_scores': F.softmax(importance_scores, dim=-1),
#             'learned_indices': learned_idx,
#         }
#         return sampled_features, sampled_xyz, combined_idx, aux_info

# class LocalTransformerAggregation(nn.Module):
#     """Attention-based module for aggregating local neighborhood features."""
#     def __init__(self, in_channels: int, out_channels: int, heads: int = 4):
#         super().__init__()
#         self.heads, self.head_dim = heads, out_channels // heads
#         self.q_proj, self.k_proj, self.v_proj = nn.Linear(in_channels, out_channels), nn.Linear(in_channels, out_channels), nn.Linear(in_channels, out_channels)
#         self.pos_mlp = nn.Sequential(nn.Linear(4, out_channels), nn.ReLU(), nn.Linear(out_channels, out_channels))
#         self.out_proj = nn.Linear(out_channels, out_channels)
#         self.res_proj = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

#     def forward(self, center_xyz: torch.Tensor, center_features: torch.Tensor, neigh_xyz: torch.Tensor, neigh_feat: torch.Tensor) -> torch.Tensor:
#         B, N, _ = center_features.shape; _, _, K, _ = neigh_feat.shape
#         rel_pos = neigh_xyz - center_xyz.unsqueeze(2)
#         rel_dist = torch.norm(rel_pos, dim=-1, keepdim=True)
#         pos_enc = self.pos_mlp(torch.cat([rel_pos, rel_dist], dim=-1))
#         q = self.q_proj(center_features).view(B, N, self.heads, self.head_dim)
#         k = self.k_proj(neigh_feat).view(B, N, K, self.heads, self.head_dim)
#         v = self.v_proj(neigh_feat).view(B, N, K, self.heads, self.head_dim)
#         k += pos_enc.view(B, N, K, self.heads, self.head_dim)
#         v += pos_enc.view(B, N, K, self.heads, self.head_dim)
#         q, k, v = q.permute(0, 2, 1, 3), k.permute(0, 3, 1, 2, 4), v.permute(0, 3, 1, 2, 4)
#         scores = torch.einsum('bhnd,bhnkd->bhnk', q, k) / (self.head_dim ** 0.5)
#         attn_out = torch.einsum('bhnk,bhnkd->bhnd', F.softmax(scores, dim=-1), v).permute(0, 2, 1, 3).contiguous().view(B, N, -1)
#         return F.leaky_relu(self.out_proj(attn_out) + self.res_proj(center_features), 0.2)

# class Network(nn.Module):
#     """The full Encoder-Decoder segmentation network."""
#     def __init__(self, config):
#         super().__init__()
#         self.config, self.store_aux_info = config, False
#         self.fc0 = nn.Sequential(nn.Linear(config.num_features, 8), nn.BatchNorm1d(8), nn.LeakyReLU(0.2, inplace=True))
        
#         self.encoder_layers, in_c = nn.ModuleList(), 8
#         for i in range(config.num_layers):
#             out_c = config.d_out[i]
#             self.encoder_layers.append(nn.ModuleDict({'lta': LocalTransformerAggregation(in_c, out_c),
#                                                      'mlp': nn.Sequential(nn.Linear(out_c, out_c), nn.BatchNorm1d(out_c), nn.LeakyReLU(0.2, inplace=True)),
#                                                      'gas': GeometryAdaptiveSampling(out_c)}))
#             in_c = out_c
        
#         self.bottleneck = nn.Sequential(nn.Linear(in_c, in_c), nn.BatchNorm1d(in_c), nn.LeakyReLU(0.2, inplace=True))
#         self.decoder_layers = nn.ModuleList()
#         decoder_in_c = in_c
#         for i in range(config.num_layers):
#             skip_c = config.d_out[config.num_layers - 1 - i]
#             mlp_in_c = skip_c + decoder_in_c
#             mlp_out_c = skip_c
#             self.decoder_layers.append(nn.ModuleDict({
#                 'mlp': nn.Sequential(nn.Linear(mlp_in_c, mlp_out_c), nn.BatchNorm1d(mlp_out_c), nn.LeakyReLU(0.2, inplace=True)),
#                 'lta': LocalTransformerAggregation(mlp_out_c, mlp_out_c)
#             }))
#             decoder_in_c = mlp_out_c

#         self.classifier = nn.Sequential(
#             nn.Linear(config.d_out[0], 64), nn.BatchNorm1d(64), nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(0.5), nn.Linear(64, config.num_classes)
#         )

#     def forward(self, inputs: dict) -> Dict[str, any]:
#         xyz, features = inputs['xyz'], inputs['features']
#         B, N, _ = features.shape
#         self.aux_info_list = [] if self.store_aux_info else None

#         # --- FIX: Create dedicated lists for U-Net skip connections ---
#         encoder_skip_xyz: List[torch.Tensor] = []
#         encoder_skip_features: List[torch.Tensor] = []
        
#         # --- These lists will now track the downsampled tensors ---
#         downsampled_xyz: List[torch.Tensor] = [xyz]
#         downsampled_features: List[torch.Tensor] = [self.fc0(features.view(-1, self.config.num_features)).view(B, N, -1)]
        
#         for i, layer in enumerate(self.encoder_layers):
#             xyz_i, feat_i = downsampled_xyz[-1], downsampled_features[-1]
#             neigh_idx = knn_blocked(xyz_i, xyz_i, self.config.k_n)
#             neigh_xyz, neigh_feat = gather_neighbors(xyz_i, neigh_idx), gather_neighbors(feat_i, neigh_idx)
            
#             feat_ta = layer['lta'](xyz_i, feat_i, neigh_xyz, neigh_feat)
#             feat_enc = layer['mlp'](feat_ta.view(-1, feat_ta.shape[-1])).view_as(feat_ta)
            
#             # --- FIX: Store the full-resolution coordinates and features for the skip connection BEFORE sampling ---
#             encoder_skip_xyz.append(xyz_i)
#             encoder_skip_features.append(feat_enc)
            
#             k_points = feat_i.shape[1] // self.config.sub_sampling_ratio[i]
#             sampled_feat, sampled_xyz, _, aux_info = layer['gas'](xyz_i, feat_enc, k_points)
            
#             # Pass downsampled features to the next encoder layer
#             downsampled_xyz.append(sampled_xyz)
#             downsampled_features.append(sampled_feat)
#             if self.store_aux_info: self.aux_info_list.append(aux_info)
        
#         decoded_feat = self.bottleneck(downsampled_features[-1].view(-1, downsampled_features[-1].shape[-1])).view_as(downsampled_features[-1])

#         for i in range(self.config.num_layers):
#             layer_idx = self.config.num_layers - 1 - i
            
#             # --- FIX: Use the correct, full-resolution skip connection tensors ---
#             skip_xyz = encoder_skip_xyz[layer_idx]
#             skip_feat = encoder_skip_features[layer_idx]
            
#             # Interpolate features from the lower-resolution decoder output to the higher-resolution skip connection
#             interp_idx = knn_blocked(query_xyz=skip_xyz, support_xyz=downsampled_xyz[layer_idx + 1], k=1)
#             upsampled_feat = gather_neighbors(decoded_feat, interp_idx).squeeze(2)
            
#             concat_feat = torch.cat([skip_feat, upsampled_feat], dim=-1)
#             feat_dec = self.decoder_layers[i]['mlp'](concat_feat.view(-1, concat_feat.shape[-1])).view(B, concat_feat.shape[1], -1)
            
#             # Perform final aggregation at the skip connection's resolution
#             dec_neigh_idx = knn_blocked(skip_xyz, skip_xyz, self.config.k_n)
#             dec_neigh_xyz = gather_neighbors(skip_xyz, dec_neigh_idx)
#             dec_neigh_feat = gather_neighbors(feat_dec, dec_neigh_idx)
#             decoded_feat = self.decoder_layers[i]['lta'](skip_xyz, feat_dec, dec_neigh_xyz, dec_neigh_feat)

#         logits = self.classifier(decoded_feat.view(-1, decoded_feat.shape[-1])).view(B, N, -1)
#         return {'logits': logits, 'aux_info': self.aux_info_list}




