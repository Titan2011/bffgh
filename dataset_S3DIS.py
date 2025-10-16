"""
A modern and clean PyTorch Dataset class for the S3DIS dataset.
This implementation completely replaces the old TensorFlow 1.x data pipeline.
It handles both training batch generation and the specific inference-time
batch generation required for full-cloud evaluation.
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pickle
from os.path import join
import logging

from helper_ply import read_ply
from helper_tool import ConfigS3DIS as cfg
from helper_tool import DataProcessing as DP

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class S3DISDataset(Dataset):
    def __init__(self, mode, test_area_idx=5):
        self.config = cfg()
        self.mode = mode
        
        base_path = self.config.data_path
        sub_pc_folder = join(base_path, 'input_{:.3f}'.format(self.config.sub_grid_size))
        
        if not os.path.exists(sub_pc_folder):
            raise FileNotFoundError(f"Data folder not found: {sub_pc_folder}")

        # Get list of all sub-sampled .ply files
        all_files = sorted([f for f in os.listdir(sub_pc_folder) if f.endswith('.ply')])

        # Split files based on test area
        val_split_name = f'Area_{test_area_idx}'
        
        self.cloud_names = []
        if self.mode == 'training':
            self.cloud_names = [f[:-4] for f in all_files if val_split_name not in f]
        elif self.mode in ['validation', 'test']:
            self.cloud_names = [f[:-4] for f in all_files if val_split_name in f]
        
        # Load all data into memory
        self.input_points, self.input_colors, self.input_labels = {'training': [], 'validation': []}, {'training': [], 'validation': []}, {'training': [], 'validation': []}
        
        # Also load data needed for full-resolution testing
        self.projection_inds, self.validation_labels = [], []

        for cloud_name in self.cloud_names:
            ply_path = join(sub_pc_folder, cloud_name + '.ply')
            data = read_ply(ply_path)
            points = np.vstack((data['x'], data['y'], data['z'])).T.astype(np.float32)
            colors = np.vstack((data['red'], data['green'], data['blue'])).T.astype(np.float32)
            labels = data['class'].astype(np.int64)

            split = 'validation' if val_split_name in cloud_name else 'training'
            self.input_points[split].append(points)
            self.input_colors[split].append(colors)
            self.input_labels[split].append(labels)
            
            # Load projection indices for validation/test set
            if split == 'validation':
                proj_file = join(sub_pc_folder, f'{cloud_name}_proj.pkl')
                with open(proj_file, 'rb') as f:
                    proj, proj_labels = pickle.load(f)
                self.projection_inds.append(proj)
                self.validation_labels.append(proj_labels)

        # Create a single large point cloud for training for efficient batching
        if self.mode == 'training':
            self.points = np.concatenate(self.input_points['training'], axis=0)
            self.colors = np.concatenate(self.input_colors['training'], axis=0)
            self.labels = np.concatenate(self.input_labels['training'], axis=0)
            
            # Spatially regular sampling for training batches
            self.possibility = np.random.rand(self.points.shape[0]) * 1e-3
            
        logging.info(f"Initialized S3DIS in '{self.mode}' mode with {len(self.cloud_names)} clouds (Test Area: {test_area_idx}).")

    def __len__(self):
        # Length is the number of steps per epoch
        if self.mode == 'training':
            return self.config.train_steps * self.config.batch_size
        else: # Validation/Test
            return self.config.val_steps * self.config.val_batch_size

    def __getitem__(self, index):
        """
        For training, generates a single point cloud patch on-the-fly.
        For validation, this is not used; see `generate_test_batches`.
        """
        if self.mode != 'training':
            # This is handled by the test function directly
            return {}

        # Choose a center point for the patch based on spatial probability
        point_ind = np.argmin(self.possibility)
        center_point = self.points[point_ind, :].reshape(1, -1)
        
        # Simple KNN search for patch
        dists = np.sum(np.square(self.points - center_point), axis=1)
        queried_idx = np.argsort(dists)[:self.config.num_points]
        
        # Update sampling possibility
        dists_queried = dists[queried_idx]
        delta = np.square(1 - dists_queried / np.max(dists_queried))
        self.possibility[queried_idx] += delta
        
        # Get patch data
        xyz = self.points[queried_idx]
        xyz = xyz - center_point # Center the patch
        colors = self.colors[queried_idx]
        labels = self.labels[queried_idx]
        
        return {
            'xyz': xyz.astype(np.float32),
            'features': colors.astype(np.float32),
            'labels': labels.astype(np.int64)
        }

    def generate_test_batches(self):
        """
        Generator for creating batches during inference/testing, where each
        batch is a sphere of points from one of the validation clouds.
        """
        for i, cloud_points in enumerate(self.input_points['validation']):
            cloud_name = self.cloud_names[i]
            # Choose a center point (can be more sophisticated, e.g., grid)
            center_point_idx = np.random.randint(0, len(cloud_points))
            center_point = cloud_points[center_point_idx].reshape(1, -1)
            
            dists = np.sum(np.square(cloud_points - center_point), axis=1)
            queried_idx = np.argsort(dists)[:self.config.num_points]
            
            xyz = cloud_points[queried_idx]
            xyz = xyz - center_point
            colors = self.input_colors['validation'][i][queried_idx]
            labels = self.input_labels['validation'][i][queried_idx]

            yield cloud_name, center_point, xyz, colors, labels

    @staticmethod
    def collate_fn(batch):
        """
        Collate function for the training DataLoader.
        """
        # Batch is already a list of dicts from __getitem__
        xyz = np.stack([s['xyz'] for s in batch])
        features = np.stack([s['features'] for s in batch])
        labels = np.stack([s['labels'] for s in batch])
        
        # Normalize colors to be between -0.5 and 0.5 for better network stability
        features = (features / 255.0) - 0.5
        
        return {
            'xyz': torch.from_numpy(xyz),
            'features': torch.from_numpy(features),
            'labels': torch.from_numpy(labels),
        }

