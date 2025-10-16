# import torch
# from torch.utils.data import Dataset
# import numpy as np
# import pickle
# from os.path import join, exists, splitext
# import os
# from helper_ply import read_ply
# from helper_tool import ConfigSemantic3D as cfg

# class Semantic3D(Dataset):
#     def __init__(self, mode):
#         """
#         Lightweight, robust dataset loader for Semantic3D.
#         - mode: 'training' or anything else interpreted as validation
#         Assumptions:
#         - Preprocessed point clouds live in self.sub_pc_folder
#         - Supported file extensions: .ply, .pkl, .npy, .npz
#         - Returned sample is a dict with keys:
#           'xyz' (FloatTensor Nx3), 'features' (FloatTensor NxF), 'labels' (LongTensor N),
#           'pc_idx' (LongTensor N), 'cloud_idx' (LongTensor N), 'file_path' (str)
#         """
#         self.name = 'Semantic3D'
#         self.path = 'data/semantic3d'
#         self.label_to_names = {0: 'unlabeled', 1: 'man-made terrain', 2: 'natural terrain', 3: 'high vegetation',
#                                4: 'low vegetation', 5: 'buildings', 6: 'hard scape', 7: 'scanning artefacts',
#                                8: 'cars'}
#         self.num_classes = len(self.label_to_names)
#         self.label_values = np.sort([k for k, v in self.label_to_names.items()])
#         self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
#         self.ignored_labels = np.sort([0])

#         self.sub_pc_folder = join(self.path, 'input_{:.3f}'.format(cfg.sub_grid_size))

#         # ensure folder exists
#         if not exists(self.sub_pc_folder):
#             raise FileNotFoundError(f"Sub-point-cloud folder not found: {self.sub_pc_folder}")

#         # gather supported files
#         supported_ext = ('.ply', '.pkl', '.npy', '.npz')
#         files = [join(self.sub_pc_folder, f) for f in os.listdir(self.sub_pc_folder)
#                  if f.lower().endswith(supported_ext)]
#         files.sort()  # deterministic ordering

#         # simple deterministic split: first 80% train, rest val
#         split_idx = int(0.8 * len(files))
#         self.mode = mode
#         self.train_files = files[:split_idx]
#         self.val_files = files[split_idx:]

#         self.data_list = self.train_files if mode == 'training' else self.val_files

#         if len(self.data_list) == 0:
#             raise RuntimeError(f"No data files found for mode={mode} in {self.sub_pc_folder}")

#     def __len__(self):
#         # Return the size of the dataset
#         return len(self.data_list)

#     def _load_file(self, path):
#         """
#         Load a point-cloud file and return dict with keys:
#         'xyz' (Nx3), optional 'colors' (Nx3), optional 'features' (NxF), optional 'labels' (N,)
#         """
#         ext = splitext(path)[1].lower()
#         if ext == '.ply':
#             plydata = read_ply(path)
#             # read_ply typically returns a dict-like object or numpy structured array
#             data = {}
#             if isinstance(plydata, dict):
#                 if all(k in plydata for k in ('x', 'y', 'z')):
#                     data['xyz'] = np.vstack((plydata['x'], plydata['y'], plydata['z'])).T
#                 elif 'points' in plydata:
#                     data['xyz'] = np.asarray(plydata['points'])
#                 # colors
#                 if all(k in plydata for k in ('red', 'green', 'blue')):
#                     data['colors'] = np.vstack((plydata['red'], plydata['green'], plydata['blue'])).T
#                 # labels
#                 for lbl_key in ('class', 'label', 'labels'):
#                     if lbl_key in plydata:
#                         data['labels'] = np.asarray(plydata[lbl_key]).astype(np.int64)
#                         break
#             else:
#                 # try structured array fields
#                 try:
#                     xyz = np.vstack((plydata['x'], plydata['y'], plydata['z'])).T
#                     data['xyz'] = xyz
#                     if 'red' in plydata.dtype.names:
#                         data['colors'] = np.vstack((plydata['red'], plydata['green'], plydata['blue'])).T
#                     if 'class' in plydata.dtype.names:
#                         data['labels'] = np.asarray(plydata['class']).astype(np.int64)
#                 except Exception:
#                     raise RuntimeError(f"Unsupported PLY format for file: {path}")
#             return data

#         elif ext == '.pkl':
#             with open(path, 'rb') as f:
#                 # Add encoding for compatibility with python 2 pickled files
#                 obj = pickle.load(f, encoding='latin1')
            
#             data = {}
#             if isinstance(obj, dict):
#                 # The object is a dictionary as expected.
#                 if 'xyz' in obj:
#                     data['xyz'] = np.asarray(obj['xyz'])
#                 if 'features' in obj:
#                     data['features'] = np.asarray(obj['features'])
#                 if 'colors' in obj:
#                     data['colors'] = np.asarray(obj['colors'])
#                 if 'labels' in obj:
#                     data['labels'] = np.asarray(obj['labels']).astype(np.int64)
            
#             elif isinstance(obj, np.ndarray):
#                 # The object is a raw numpy array, which caused the error.
#                 # Assume a structure, e.g., columns are x,y,z,[features...],[label]
#                 if obj.shape[1] < 3:
#                      raise ValueError(f"Numpy array in {path} has fewer than 3 columns.")
#                 data['xyz'] = obj[:, :3]
#                 if obj.shape[1] > 3:
#                     # Heuristically check if the last column is integer labels
#                     if np.issubdtype(obj[:, -1].dtype, np.integer):
#                         data['labels'] = obj[:, -1].astype(np.int64)
#                         if obj.shape[1] > 4:
#                             data['features'] = obj[:, 3:-1]
#                     else: # Assume no labels, rest are features
#                         data['features'] = obj[:, 3:]
#             else:
#                 raise TypeError(f"Unsupported data type {type(obj)} found in pickle file: {path}")
#             return data

#         elif ext in ('.npy', '.npz'):
#             loaded = np.load(path, allow_pickle=True)
#             data = {}
#             if ext == '.npy':
#                 arr = np.asarray(loaded)
#                 if arr.ndim == 2 and arr.shape[1] >= 3:
#                     data['xyz'] = arr[:, :3]
#                     if arr.shape[1] > 3:
#                         data['features'] = arr[:, 3:]
#                 else:
#                     raise RuntimeError(f"Unsupported .npy content for file: {path}")
#             else:  # .npz
#                 if 'xyz' in loaded:
#                     data['xyz'] = loaded['xyz']
#                 if 'features' in loaded:
#                     data['features'] = loaded['features']
#                 if 'colors' in loaded:
#                     data['colors'] = loaded['colors']
#                 if 'labels' in loaded:
#                     data['labels'] = loaded['labels'].astype(np.int64)
#             return data

#         else:
#             raise RuntimeError(f"Unsupported file extension: {ext}")

#     def __getitem__(self, idx):
#         # Load and process data
#         file_path = self.data_list[idx]
#         data = self._load_file(file_path)

#         if 'xyz' not in data:
#             raise RuntimeError(f"No xyz found in file: {file_path}")
#         xyz = np.asarray(data['xyz'], dtype=np.float32)
#         N = xyz.shape[0]

#         # features priority: explicit 'features' -> colors -> fall back to zeros
#         if 'features' in data:
#             features = np.asarray(data['features'], dtype=np.float32)
#             if features.ndim == 1:
#                 features = features[:, None]
#         elif 'colors' in data:
#             colors = np.asarray(data['colors'], dtype=np.float32)
#             # normalize color if in 0-255
#             if colors.max() > 1.0:
#                 colors = colors / 255.0
#             features = colors
#         else:
#             # fallback: use zeros (or you could use xyz coordinates as features)
#             features = np.zeros((N, 3), dtype=np.float32)

#         # labels: if missing, set to -1 (ignored)
#         if 'labels' in data:
#             labels = np.asarray(data['labels'], dtype=np.int64)
#         else:
#             labels = -1 * np.ones((N,), dtype=np.int64)

#         # pc_idx: point index within the cloud
#         pc_idx = np.arange(N, dtype=np.int64)
#         # cloud_idx: index of cloud in dataset (use idx)
#         cloud_idx = np.full((N,), idx, dtype=np.int64)

#         sample = {
#             'xyz': torch.from_numpy(xyz).float(),      # Nx3
#             'features': torch.from_numpy(features).float(),  # NxF (F may be 0)
#             'labels': torch.from_numpy(labels).long(),       # N
#             'pc_idx': torch.from_numpy(pc_idx).long(),       # N
#             'cloud_idx': torch.from_numpy(cloud_idx).long(),   # N
#             'file_path': file_path
#         }
#         return sample

#     def collate_fn(self, batch):
#         """
#         Collates a list of samples into a single batch dictionary.
#         - Samples or pads points to a fixed number (cfg.num_points).
#         - Stacks samples into batch tensors.
#         """
#         try:
#             # Number of points to sample/pad to for each cloud
#             num_points = cfg.num_points
#         except AttributeError:
#             # Fallback if not defined in config, though it should be.
#             # 40960 is a common value for RandLA-Net on Semantic3D.
#             num_points = 40960

#         stacked_xyz, stacked_features, stacked_labels = [], [], []

#         for sample in batch:
#             xyz = sample['xyz']
#             features = sample['features']
#             labels = sample['labels']
            
#             n_points = xyz.shape[0]
            
#             if n_points > num_points:
#                 # Randomly sample if there are more points than needed
#                 choice_indices = torch.randperm(n_points)[:num_points]
#             else:
#                 # Pad with duplicates if there are fewer points
#                 choice_indices = torch.arange(n_points)
#                 if n_points < num_points:
#                     # Choose random points to duplicate
#                     extra_indices = torch.randint(0, n_points, (num_points - n_points,))
#                     choice_indices = torch.cat([choice_indices, extra_indices])
            
#             stacked_xyz.append(xyz[choice_indices])
#             stacked_features.append(features[choice_indices])
#             stacked_labels.append(labels[choice_indices])
        
#         # Return a dictionary of stacked tensors
#         return {
#             'xyz': torch.stack(stacked_xyz),
#             'features': torch.stack(stacked_features),
#             'labels': torch.stack(stacked_labels),
#         }

#     def tf_map(self, batch_xyz, batch_features, batch_labels, batch_pc_idx, batch_cloud_idx):
#         """
#         Accepts either lists of numpy arrays (one per cloud) or already concatenated numpy arrays.
#         Returns a dict of torch tensors suitable for model input.
#         This mirrors a lightweight adaptation of a typical tf_map used in TensorFlow pipelines.
#         """
#         # Helper to concatenate if input is list-like
#         def _concat(x):
#             if isinstance(x, list) or isinstance(x, tuple):
#                 return np.concatenate(x, axis=0)
#             return x

#         batch_xyz = _concat(batch_xyz)
#         batch_features = _concat(batch_features) if batch_features is not None and (isinstance(batch_features, (list, tuple)) or np.ndim(batch_features) != 2) else batch_features
#         batch_labels = _concat(batch_labels) if batch_labels is not None else batch_labels
#         batch_pc_idx = _concat(batch_pc_idx)
#         batch_cloud_idx = _concat(batch_cloud_idx)

#         # Ensure numpy arrays
#         batch_xyz = np.asarray(batch_xyz, dtype=np.float32)
#         if batch_features is None:
#             # create empty features with shape (N,0)
#             batch_features = np.zeros((batch_xyz.shape[0], 0), dtype=np.float32)
#         else:
#             batch_features = np.asarray(batch_features, dtype=np.float32)
#         if batch_labels is None:
#             batch_labels = -1 * np.ones((batch_xyz.shape[0],), dtype=np.int64)
#         else:
#             batch_labels = np.asarray(batch_labels, dtype=np.int64)
#         batch_pc_idx = np.asarray(batch_pc_idx, dtype=np.int64)
#         batch_cloud_idx = np.asarray(batch_cloud_idx, dtype=np.int64)

#         return {
#             'xyz': torch.from_numpy(batch_xyz).float(),
#             'features': torch.from_numpy(batch_features).float(),
#             'labels': torch.from_numpy(batch_labels).long(),
#             'pc_idx': torch.from_numpy(batch_pc_idx).long(),
#             'cloud_idx': torch.from_numpy(batch_cloud_idx).long()
#         }