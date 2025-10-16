"""
Full PyTorch Training, Evaluation, and Analysis Pipeline for 3D Segmentation.

This script provides a complete, publication-ready framework for training and 
evaluating the RandLA-Net++ model. It includes:

1.  A standard, weighted cross-entropy loss function.
2.  A training loop with an AdamW optimizer and learning rate scheduler.
3.  A standard evaluation function to compute overall accuracy and mIoU.
4.  A specialized `evaluate_boundary_iou` function to rigorously measure
    segmentation performance specifically at class boundaries, providing the 
    core metrics needed to validate the Geometry-Adaptive Sampling (GAS) method.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from os.path import join, exists
from os import makedirs
from sklearn.metrics import confusion_matrix
from helper_tool import ConfigSemantic3D as config

from RandLANet import Network, knn_blocked


def compute_loss(logits: torch.Tensor, labels: torch.Tensor, config) -> torch.Tensor:
    """
    Computes the weighted cross-entropy loss, ignoring specified labels.
    Logically identical to the original TensorFlow `_build_loss`.
    
    Args:
        logits: Tensor of shape (B, N, num_classes).
        labels: Tensor of shape (B, N).
        config: A configuration object with `num_classes`, `ignored_label_inds`, and `class_weights`.
        
    Returns:
        The computed scalar loss.
    """
    B, N, C = logits.shape
    
    # Flatten logits and labels
    flat_logits = logits.view(-1, C)
    flat_labels = labels.view(-1)
    
    # Create class weights tensor
    class_weights = torch.tensor(config.class_weights, dtype=torch.float32, device=logits.device)

    
    loss_func = nn.CrossEntropyLoss(weight=class_weights, ignore_index=config.ignored_label_inds[0])
    loss = loss_func(flat_logits, flat_labels)
    
    return loss

def get_optimizer_and_scheduler(model: nn.Module, config):
    """
    Sets up the AdamW optimizer and a learning rate scheduler.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    # A common choice is a step-based or cosine annealing scheduler.
    # This example uses a step scheduler, equivalent to the original TF decay logic.
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_decay_step, gamma=config.lr_decay_gamma)
    
    return optimizer, scheduler

# #####################################################################################
# ######################### Core Evaluation Metrics ###################################
# #####################################################################################

@torch.no_grad()
def evaluate_standard(model: nn.Module, dataloader, config):
    """
    Standard evaluation loop to compute overall accuracy and mean IoU.
    
    Args:
        model: The trained PyTorch model.
        dataloader: PyTorch DataLoader for the validation set.
        config: Configuration object.
    
    Returns:
        A dictionary containing 'accuracy' and 'mean_iou'.
    """
    model.eval()
    all_preds = []
    all_labels = []

    for batch in dataloader:
        # Move data to the appropriate device (e.g., CUDA)
        for key, val in batch.items():
            if isinstance(val, list):
                batch[key] = [v.to(config.device) for v in val]
            else:
                batch[key] = val.to(config.device)
        
        outputs = model(batch)
        logits = outputs['logits'] # (B, N, C)
        labels = batch['labels']   # (B, N)
        
        preds = torch.argmax(logits, dim=2)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0).ravel()
    all_labels = np.concatenate(all_labels, axis=0).ravel()
    
    # Create mask to ignore specified labels
    mask = np.ones_like(all_labels, dtype=bool)
    for ign_label in config.ignored_label_inds:
        mask &= (all_labels != ign_label)
        
    valid_preds = all_preds[mask]
    valid_labels = all_labels[mask]

    # Compute confusion matrix
    cm = confusion_matrix(valid_labels, valid_preds, labels=np.arange(config.num_classes))
    
    # Compute IoU
    intersection = np.diag(cm)
    union = np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm)
    iou = intersection / (union + 1e-6) # Add epsilon to avoid division by zero
    mean_iou = np.nanmean(iou) # Use nanmean to ignore classes not present in validation set
    
    accuracy = np.sum(intersection) / (np.sum(cm) + 1e-6)
    
    return {'accuracy': accuracy, 'mean_iou': mean_iou, 'class_iou': iou}

@torch.no_grad()
def evaluate_boundary_iou(model: nn.Module, dataloader, config, k_neighbors=10):
    """
    **Crucial function for the research paper.**
    Calculates IoU specifically on points that lie on class boundaries.
    
    Args:
        model: The trained PyTorch model.
        dataloader: PyTorch DataLoader for the validation set.
        config: Configuration object.
        k_neighbors: Number of neighbors to check for identifying boundary points.
        
    Returns:
        A dictionary containing 'boundary_mean_iou'.
    """
    model.eval()
    total_boundary_cm = np.zeros((config.num_classes, config.num_classes), dtype=np.int64)
    
    for batch in dataloader:
        for key, val in batch.items():
            if isinstance(val, list):
                batch[key] = [v.to(config.device) for v in val]
            else:
                batch[key] = val.to(config.device)

        outputs = model(batch)
        logits = outputs['logits'] # (B, N, C)
        labels = batch['labels']   # (B, N)
        xyz = batch['xyz'][0]      # (B, N, 3)
        
        preds = torch.argmax(logits, dim=2)

        # Iterate over each point cloud in the batch
        for i in range(xyz.shape[0]):
            pcd_xyz = xyz[i]       # (N, 3)
            pcd_labels = labels[i] # (N)
            pcd_preds = preds[i]   # (N)

            # 1. Find K-Nearest Neighbors for the entire point cloud
            # Use the efficient knn_blocked, but on a single point cloud (add batch dim)
            neigh_idx = knn_blocked(pcd_xyz.unsqueeze(0), k=k_neighbors).squeeze(0) # (N, k)
            
            # 2. Gather labels of neighbors
            neighbor_labels = _gather_neighbors(pcd_labels.unsqueeze(0).unsqueeze(-1), neigh_idx.unsqueeze(0)).squeeze(0).squeeze(-1) # (N, k)
            
            # 3. Identify boundary points: A point is a boundary point if its label is
            #    different from any of its neighbors' labels.
            is_boundary = torch.any(neighbor_labels != pcd_labels.unsqueeze(1), dim=1) # (N)
            
            boundary_labels_cpu = pcd_labels[is_boundary].cpu().numpy()
            boundary_preds_cpu = pcd_preds[is_boundary].cpu().numpy()

            # 4. Mask out ignored labels from the boundary points
            mask = np.ones_like(boundary_labels_cpu, dtype=bool)
            for ign_label in config.ignored_label_inds:
                mask &= (boundary_labels_cpu != ign_label)

            valid_boundary_labels = boundary_labels_cpu[mask]
            valid_boundary_preds = boundary_preds_cpu[mask]

            # 5. Accumulate confusion matrix for boundary points
            if valid_boundary_labels.size > 0:
                total_boundary_cm += confusion_matrix(
                    valid_boundary_labels, 
                    valid_boundary_preds, 
                    labels=np.arange(config.num_classes)
                )

    # Compute IoU from the accumulated boundary confusion matrix
    intersection = np.diag(total_boundary_cm)
    union = np.sum(total_boundary_cm, axis=1) + np.sum(total_boundary_cm, axis=0) - np.diag(total_boundary_cm)
    iou = intersection / (union + 1e-6)
    boundary_mean_iou = np.nanmean(iou)

    return {'boundary_mean_iou': boundary_mean_iou, 'boundary_class_iou': iou}

# #####################################################################################
# ######################### Main Training and Validation Loop #########################
# #####################################################################################

def train_one_epoch(model, dataloader, optimizer, config):
    """A single training epoch."""
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        for key, val in batch.items():
            if isinstance(val, list):
                batch[key] = [v.to(config.device) for v in val]
            else:
                batch[key] = val.to(config.device)
        
        outputs = model(batch)
        loss = compute_loss(outputs['logits'], batch['labels'], config)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def main():
    """Main function to run the training and evaluation pipeline."""
    # --- 1. Configuration (Replace with your actual config) ---
    # class Config:
    #     num_classes = 13
    #     num_features = 6 
    #     class_weights = [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 1.0, 3.0, 3.0, 3.0, 3.0]
    #     ignored_label_inds = [0]
        
    #     # Model Hyperparameters
    #     d_out = [16, 64, 128, 256]
    #     num_layers = 4
    #     k_n = 16
    #     sub_sampling_ratio = [4, 4, 4, 4]
        
    #     # Training Hyperparameters
    #     learning_rate = 1e-3
    #     weight_decay = 1e-4
    #     lr_decay_step = 5
    #     lr_decay_gamma = 0.95
    #     max_epoch = 100
    #     device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    #     # Paths
    #     saving_path = 'snapshots'

    # config = Config()
    
    # --- 2. DataLoaders (Replace with your actual DataLoaders) ---
    # This is a placeholder. You need to implement your own PyTorch Dataset and DataLoader.
    print("Using mock DataLoaders. Replace with your actual data pipeline.")
    from torch.utils.data import DataLoader, TensorDataset
    # Mock data: B=2, N=4096 points
    mock_xyz = torch.randn(10, 4096, 3)
    mock_features = torch.randn(10, 4096, config.num_features)
    mock_labels = torch.randint(0, config.num_classes, (10, 4096))
    mock_interp = torch.zeros(10, 4096, 1, dtype=torch.long)
    mock_dataset = TensorDataset(mock_xyz, mock_features, mock_labels, mock_interp)
    
    # In a real scenario, your Dataset's __getitem__ would return a dictionary
    def collate_fn(batch):
        xyz, features, labels, interp = zip(*batch)
        return {
            'xyz': [torch.stack(xyz)], 
            'features': torch.stack(features), 
            'labels': torch.stack(labels),
            'interp_idx': [interp[0]] * config.num_layers # Mocking interp_idx
        }
    
    train_loader = DataLoader(mock_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(mock_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    # --- 3. Model, Optimizer, Loss ---
    model = Network(config).to(config.device)
    optimizer, scheduler = get_optimizer_and_scheduler(model, config)
    
    # --- 4. Training and Evaluation Loop ---
    best_boundary_iou = 0.0
    
    for epoch in range(config.max_epoch):
        start_time = time.time()
        
        # Training
        avg_loss = train_one_epoch(model, train_loader, optimizer, config)
        
        # Validation
        standard_metrics = evaluate_standard(model, val_loader, config)
        boundary_metrics = evaluate_boundary_iou(model, val_loader, config)
        
        scheduler.step()
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{config.max_epoch} | Time: {epoch_time:.2f}s | Loss: {avg_loss:.4f} | "
              f"mIoU: {standard_metrics['mean_iou']:.4f} | "
              f"Boundary mIoU: {boundary_metrics['boundary_mean_iou']:.4f}")
              
        # Save best model based on boundary IoU
        if boundary_metrics['boundary_mean_iou'] > best_boundary_iou:
            best_boundary_iou = boundary_metrics['boundary_mean_iou']
            print(f"New best boundary mIoU: {best_boundary_iou:.4f}. Saving model...")
            if not exists(config.saving_path):
                makedirs(config.saving_path)
            save_path = join(config.saving_path, 'best_model.pth')
            torch.save(model.state_dict(), save_path)
            
    print("Training finished.")
    print(f"Best Boundary mIoU achieved: {best_boundary_iou:.4f}")

if __name__ == '__main__':
    main()
