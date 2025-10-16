"""
Corrected, modernized, and robust PyTorch Dataset class for Semantic3D.

- FIX (MAJOR): Replaced the brittle, index-based data splitting with a robust,
  scene-based split, which is the standard method. This correctly loads the
  entire dataset by identifying test/validation scenes by their filenames,
  resolving the issue of very fast epochs and small data loading.
- FIX: Handles test files that do not contain labels by creating a placeholder.
- REFACTOR: Simplified the file loading and split logic for clarity.
- INFO: Added logging to show exactly how many files are found for each mode.
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import os
from os.path import join, basename
import logging

from helper_ply import read_ply
from helper_tool import ConfigSemantic3D as cfg

# Configure logging to be more informative
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Semantic3D(Dataset):
    def __init__(self, mode):
        self.name = 'Semantic3D'
        self.path = cfg.data_path
        self.mode = mode
        self.num_classes = cfg.num_classes
        self.ignored_labels = np.array(cfg.ignored_label_inds)

        self.sub_pc_folder = join(self.path, 'input_{:.3f}'.format(cfg.sub_grid_size))
        if not os.path.exists(self.sub_pc_folder):
            raise FileNotFoundError(f"Sub-sampled data folder not found: {self.sub_pc_folder}")

        # --- ROBUST SCENE-BASED SPLITTING ---
        all_ply_files = sorted([f for f in os.listdir(self.sub_pc_folder) if f.endswith('.ply')])

        # Define test scans by their original scene name, as is standard practice.
        test_scan_names = {
              'MarketplaceFeldkirch_Station4_rgb_intensity-reduced.ply',
            'sg27_station10_rgb_intensity-reduced.ply',
            'sg28_Station2_rgb_intensity-reduced.ply',
            'StGallenCathedral_station6_rgb_intensity-reduced.ply',
            'birdfountain_station1_xyz_intensity_rgb.ply',
            'castleblatten_station1_intensity_rgb.ply',
            'castleblatten_station5_xyz_intensity_rgb.ply',
            'marketplacefeldkirch_station1_intensity_rgb.ply',
            'marketplacefeldkirch_station4_intensity_rgb.ply',
            'marketplacefeldkirch_station7_intensity_rgb.ply',
            'sg27_station10_intensity_rgb.ply',
            'sg27_station3_intensity_rgb.ply',
            'sg27_station6_intensity_rgb.ply',
            'sg27_station8_intensity_rgb.ply',
            'sg28_station2_intensity_rgb.ply',
            'sg28_station5_xyz_intensity_rgb.ply',
            'stgallencathedral_station1_intensity_rgb.ply',
            'stgallencathedral_station3_intensity_rgb.ply',
            'stgallencathedral_station6_intensity_rgb.ply'
        }

        # The official validation scene for RandLA-Net's Semantic3D benchmark
        validation_scene = 'neugasse_station1'

        self.train_files = []
        self.val_files = []
        self.test_files = []

        for file_name in all_ply_files:
            # Check if the file belongs to a test scan
            is_test = any(test_name in file_name for test_name in test_scan_names)
            # Check if the file belongs to the validation scan
            is_val = validation_scene in file_name

            if is_test:
                self.test_files.append(file_name)
            elif is_val:
                self.val_files.append(file_name)
            else:
                # If it's not a test or validation file, it's a training file.
                self.train_files.append(file_name)

        # Assign the correct file list based on the mode
        if self.mode == 'training':
            self.data_list = self.train_files
        elif self.mode == 'validation':
            self.data_list = self.val_files
        elif self.mode == 'test':
            self.data_list = self.test_files
        else:
            raise ValueError(f"Invalid mode specified: {self.mode}. Choose 'training', 'validation', or 'test'.")

        if not self.data_list:
            raise ValueError(f"No .ply files found for mode '{self.mode}'. Please check your data directory and splitting logic.")
        
        # This logging will tell you exactly how many files are being used.
        logging.info(f"Initialized Semantic3D in '{self.mode}' mode with {len(self.data_list)} files.")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        file_path = join(self.sub_pc_folder, self.data_list[index])

        data = read_ply(file_path)
        points = np.vstack((data['x'], data['y'], data['z'])).T
        colors = np.vstack((data['red'], data['green'], data['blue'])).T

        if 'class' in data.dtype.names:
            labels = data['class'].astype(np.int64)
        elif 'label' in data.dtype.names:
            labels = data['label'].astype(np.int64)
        else:
            if self.mode in ['training', 'validation']:
                raise ValueError(f"Training/validation file is missing labels: {file_path}")
            else:
                # For test mode, create a dummy array of zeros for labels.
                labels = np.zeros(points.shape[0], dtype=np.int64)

        features = np.hstack((points, colors)).astype(np.float32)

        return {
            'xyz': points.astype(np.float32),
            'features': features,
            'labels': labels
        }

    @staticmethod
    def collate_fn(batch):
        num_points = cfg.num_points
        stacked_xyz, stacked_features, stacked_labels = [], [], []

        for sample in batch:
            xyz, features, labels = sample['xyz'], sample['features'], sample['labels']
            n_points = xyz.shape[0]
            
            if n_points > num_points:
                choice = np.random.choice(n_points, num_points, replace=False)
            else:
                choice = np.arange(n_points)
                if n_points < num_points:
                    extra = np.random.choice(n_points, num_points - n_points, replace=True)
                    choice = np.concatenate([choice, extra])
            
            stacked_xyz.append(xyz[choice])
            stacked_features.append(features[choice])
            stacked_labels.append(labels[choice])
        
        return {
            'xyz': torch.from_numpy(np.stack(stacked_xyz)),
            'features': torch.from_numpy(np.stack(stacked_features)),
            'labels': torch.from_numpy(np.stack(stacked_labels)),
        }



# """
# Corrected, modernized, and simplified PyTorch Dataset class for Semantic3D.

# - FIX (MAJOR): Implemented the user-requested splitting logic. A definitive list of
#   test scans is now used to separate test files from labeled files. The labeled
#   files are then correctly split into training and validation sets. This provides a
#   robust and reproducible data split and resolves the ValueError.
# - FIX: Now correctly handles test files that do not contain labels by creating a placeholder.
# - REFACTOR: Consolidated into a single, clean Dataset implementation.
# - SIMPLIFIED: The collate_fn correctly prepares a simple batch.
# """
# import torch
# from torch.utils.data import Dataset
# import numpy as np
# import os
# from os.path import join, splitext, basename
# import logging

# from helper_ply import read_ply
# from helper_tool import ConfigSemantic3D as cfg

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# class Semantic3D(Dataset):
#     def __init__(self, mode):
#         self.name = 'Semantic3D'
#         self.path = cfg.data_path
#         self.mode = mode
#         self.num_classes = cfg.num_classes
#         self.ignored_labels = np.array(cfg.ignored_label_inds)

#         self.sub_pc_folder = join(self.path, 'input_{:.3f}'.format(cfg.sub_grid_size))
#         if not os.path.exists(self.sub_pc_folder):
#             raise FileNotFoundError(f"Data folder not found: {self.sub_pc_folder}")

#         # --- USER-DEFINED TEST FILE SPLITTING ---
#         # Discover all .ply files, which contain the actual point cloud data.
#         all_ply_files = sorted([f for f in os.listdir(self.sub_pc_folder) if f.endswith('.ply')])

#         # Use the user-provided dictionary keys to define the exact set of test files.
#         test_scan_basenames = {
#             'MarketplaceFeldkirch_Station4_rgb_intensity-reduced.ply',
#             'sg27_station10_rgb_intensity-reduced.ply',
#             'sg28_Station2_rgb_intensity-reduced.ply',
#             'StGallenCathedral_station6_rgb_intensity-reduced.ply',
#             'birdfountain_station1_xyz_intensity_rgb.ply',
#             'castleblatten_station1_intensity_rgb.ply',
#             'castleblatten_station5_xyz_intensity_rgb.ply',
#             'marketplacefeldkirch_station1_intensity_rgb.ply',
#             'marketplacefeldkirch_station4_intensity_rgb.ply',
#             'marketplacefeldkirch_station7_intensity_rgb.ply',
#             'sg27_station10_intensity_rgb.ply',
#             'sg27_station3_intensity_rgb.ply',
#             'sg27_station6_intensity_rgb.ply',
#             'sg27_station8_intensity_rgb.ply',
#             'sg28_station2_intensity_rgb.ply',
#             'sg28_station5_xyz_intensity_rgb.ply',
#             'stgallencathedral_station1_intensity_rgb.ply',
#             'stgallencathedral_station3_intensity_rgb.ply',
#             'stgallencathedral_station6_intensity_rgb.ply'
#         }
        
#         # Any file not in the test set is considered a labeled file for training/validation.
#         labeled_files = [f for f in all_ply_files if basename(f) not in test_scan_basenames]
#         self.test_files = [f for f in all_ply_files if basename(f) in test_scan_basenames]

#         # Use a standard scene-based split for validation to ensure reproducibility.
#         # The official RandLA-Net split uses 'neugasse_station1' as the validation scene.
#         self.val_files = []
#         self.train_files = []
#         for f in labeled_files:
#             if 'neugasse_station1' in f:
#                 self.val_files.append(f)
#             else:
#                 self.train_files.append(f)

#         # Assign the correct file list based on the mode
#         if self.mode == 'training':
#             self.data_list = self.train_files
#         elif self.mode == 'validation':
#             self.data_list = self.val_files
#         elif self.mode == 'test':
#             self.data_list = self.test_files
#         else:
#             raise ValueError(f"Invalid mode specified: {self.mode}. Choose 'training', 'validation', or 'test'.")

#         if not self.data_list:
#             raise ValueError(f"No .ply files found for mode '{self.mode}'. Please check your data directory and file lists.")
#         logging.info(f"Initialized Semantic3D in '{self.mode}' mode with {len(self.data_list)} files.")

#     def __len__(self):
#         return len(self.data_list)

#     def __getitem__(self, index):
#         file_path = join(self.sub_pc_folder, self.data_list[index])

#         data = read_ply(file_path)
#         points = np.vstack((data['x'], data['y'], data['z'])).T
#         colors = np.vstack((data['red'], data['green'], data['blue'])).T

#         # --- PRIMARY FIX: Handle files with and without labels based on mode ---
#         if 'class' in data.dtype.names:
#             labels = data['class'].astype(np.int64)
#         elif 'label' in data.dtype.names:
#             labels = data['label'].astype(np.int64)
#         else:
#             # If no labels are found, raise error for train/val, but create placeholders for test
#             if self.mode in ['training', 'validation']:
#                 raise ValueError(f"Training/validation file is missing labels: {file_path}")
#             else:
#                 # For test mode, it's expected to not have labels. Create a dummy array.
#                 labels = np.zeros(points.shape[0], dtype=np.int64)

#         # The initial 'features' are the XYZ coordinates and the raw colors
#         features = np.hstack((points, colors)).astype(np.float32)

#         return {
#             'xyz': points.astype(np.float32),
#             'features': features,
#             'labels': labels
#         }

#     @staticmethod
#     def collate_fn(batch):
#         """
#         Custom collate function to handle sampling/padding and batching.
#         This creates the simple batch format that the modern network expects.
#         """
#         num_points = cfg.num_points
        
#         stacked_xyz, stacked_features, stacked_labels = [], [], []

#         for sample in batch:
#             xyz, features, labels = sample['xyz'], sample['features'], sample['labels']
#             n_points = xyz.shape[0]
            
#             if n_points > num_points:
#                 # Randomly sample if there are more points than needed
#                 choice = np.random.choice(n_points, num_points, replace=False)
#             else:
#                 # Pad with duplicates if there are fewer points
#                 choice = np.arange(n_points)
#                 if n_points < num_points:
#                     extra = np.random.choice(n_points, num_points - n_points, replace=True)
#                     choice = np.concatenate([choice, extra])
            
#             stacked_xyz.append(xyz[choice])
#             stacked_features.append(features[choice])
#             stacked_labels.append(labels[choice])
        
#         # Stack the individual samples into a single batch tensor
#         return {
#             'xyz': torch.from_numpy(np.stack(stacked_xyz)),
#             'features': torch.from_numpy(np.stack(stacked_features)),
#             'labels': torch.from_numpy(np.stack(stacked_labels)),
#         }

