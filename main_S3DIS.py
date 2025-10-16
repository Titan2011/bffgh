"""
A complete, modern PyTorch training script for the S3DIS dataset, designed to work
with the custom Geometry-Adaptive Sampling (GAS) network architecture.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os
import logging

# --- Import project modules ---
# Assumes these files are in the same directory
from helper_tool import ConfigS3DIS as cfg
from helper_tool import DataProcessing as DP
from RandLANet import Network 
from dataset_S3DIS import S3DISDataset
from boundary_metrics import BoundaryAwareMetrics 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_dataloader(mode: str, config, test_area_idx: int):
    """Initializes and returns a DataLoader for the specified mode."""
    dataset = S3DISDataset(mode, test_area_idx=test_area_idx)
    return DataLoader(
        dataset,
        batch_size=config.batch_size if mode == 'training' else config.val_batch_size,
        shuffle=(mode == 'training'),
        num_workers=getattr(config, 'num_workers', 4),
        collate_fn=dataset.collate_fn
    )

def train(FLAGS):
    """Main training and validation loop for S3DIS."""
    config = cfg()
    device = torch.device(f"cuda:{FLAGS.gpu}" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    logging.info(f"Training with Test Area: {FLAGS.test_area}")

    # Initialize DataLoaders
    train_loader = get_dataloader('training', config, test_area_idx=FLAGS.test_area)
    val_loader = get_dataloader('validation', config, test_area_idx=FLAGS.test_area)
    
    # Initialize Model, Optimizer, and Loss
    model = Network(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_decays.get(0, 5), gamma=0.95)

    # S3DIS does not have ignored labels by default, but we can set it for robustness
    class_weights = torch.from_numpy(DP.get_class_weights('S3DIS').squeeze()).float().to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    best_boundary_miou = 0.0
    for epoch in range(config.max_epoch):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.max_epoch}", leave=False):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch)
            loss = loss_fn(outputs['logits'].transpose(1, 2), batch['labels'])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1} - Average Training Loss: {avg_loss:.4f}")

        # --- Validation Loop ---
        if (epoch + 1) % 5 == 0 or epoch == config.max_epoch - 1:
            model.eval()
            metrics_computer = BoundaryAwareMetrics(config.num_classes)
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                           batch[k] = v.to(device)
                    outputs = model(batch)
                    metrics_computer.update(outputs['logits'], batch['labels'], batch['xyz'])
            
            metrics = metrics_computer.get_metrics()
            logging.info(f"\nValidation Epoch {epoch+1} - mIoU: {metrics['mIoU']:.4f}, Boundary mIoU: {metrics['boundary_mIoU']:.4f}")
            
            if metrics['boundary_mIoU'] > best_boundary_miou:
                best_boundary_miou = metrics['boundary_mIoU']
                logging.info(f"*** New best model! Saving... Best Boundary mIoU: {best_boundary_miou:.4f} ***")
                
                log_dir = getattr(config, 'log_dir', 'logs')
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                save_path = os.path.join(log_dir, f's3dis_area{FLAGS.test_area}_best_model.pth')
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID to use')
    parser.add_argument('--test_area', type=int, default=5, choices=[1, 2, 3, 4, 5, 6], help='Which area to use for test/validation')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Mode: train or test')
    parser.add_argument('--model_path', type=str, default=None, help='Path to a trained model checkpoint for testing')
    FLAGS = parser.parse_args()
    
    if FLAGS.mode == 'train':
        train(FLAGS)
    








# from os.path import join, exists, dirname, abspath, normpath, basename
# from RandLANet import Network
# from tester_S3DIS import ModelTester
# from helper_ply import read_ply
# from helper_tool import ConfigS3DIS as cfg
# from helper_tool import DataProcessing as DP
# from helper_tool import Plot
# import tensorflow as tf
# import numpy as np
# import time, pickle, argparse, glob, os

# class S3DIS:
#     def __init__(self, test_area_idx):
#         self.name = 'S3DIS'
#         # Use a relative path; normpath will convert to the proper separator
#         self.path = normpath('./data/S3DIS')
#         self.label_to_names = {0: 'ceiling',
#                                1: 'floor',
#                                2: 'wall',
#                                3: 'beam',
#                                4: 'column',
#                                5: 'window',
#                                6: 'door',
#                                7: 'table',
#                                8: 'chair',
#                                9: 'sofa',
#                                10: 'bookcase',
#                                11: 'board',
#                                12: 'clutter'}
#         self.num_classes = len(self.label_to_names)
#         self.label_values = np.sort([k for k, v in self.label_to_names.items()])
#         self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
#         self.ignored_labels = np.array([])

#         self.val_split = 'Area_' + str(test_area_idx)
#         self.all_files = glob.glob(join(self.path, 'original_ply', '*.ply'))

#         # Initiate containers
#         self.val_proj = []
#         self.val_labels = []
#         self.possibility = {}
#         self.min_possibility = {}
#         self.input_trees = {'training': [], 'validation': []}
#         self.input_colors = {'training': [], 'validation': []}
#         self.input_labels = {'training': [], 'validation': []}
#         self.input_names = {'training': [], 'validation': []}
#         self.load_sub_sampled_clouds(cfg.sub_grid_size)

#     def load_sub_sampled_clouds(self, sub_grid_size):
#         tree_path = join(self.path, 'input_{:.3f}'.format(sub_grid_size))
#         # Normalize tree_path
#         tree_path = normpath(tree_path)
#         for i, file_path in enumerate(self.all_files):
#             # Normalize the file path
#             file_path = normpath(file_path)
#             # Use basename to extract the file name without extension
#             cloud_name = basename(file_path)[:-4]
#             print("Checking file:", file_path, "-> cloud_name:", cloud_name)
#             if self.val_split in cloud_name:
#                 cloud_split = 'validation'
#             else:
#                 cloud_split = 'training'

#             # Name of the input files using join (backslashes will be inserted automatically on Windows)
#             kd_tree_file = join(tree_path, '{}_KDTree.pkl'.format(cloud_name))
#             sub_ply_file = join(tree_path, '{}.ply'.format(cloud_name))

#             data = read_ply(sub_ply_file)
#             sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
#             sub_labels = data['class']

#             # Read pkl with search tree
#             with open(kd_tree_file, 'rb') as f:
#                 search_tree = pickle.load(f)

#             self.input_trees[cloud_split] += [search_tree]
#             self.input_colors[cloud_split] += [sub_colors]
#             self.input_labels[cloud_split] += [sub_labels]
#             self.input_names[cloud_split] += [cloud_name]

#             size = sub_colors.shape[0] * 4 * 7
#             print('{} {:.1f} MB loaded in {:.1f}s'.format(basename(kd_tree_file), size * 1e-6, time.time() - time.time()))

#         print('\nPreparing reprojected indices for testing')
#         for i, file_path in enumerate(self.all_files):
#             file_path = normpath(file_path)
#             cloud_name = basename(file_path)[:-4]

#             if self.val_split in cloud_name:
#                 proj_file = join(tree_path, '{}_proj.pkl'.format(cloud_name))
#                 with open(proj_file, 'rb') as f:
#                     proj_idx, labels = pickle.load(f)
#                 self.val_proj += [proj_idx]
#                 self.val_labels += [labels]
#                 print('{} done in {:.1f}s'.format(cloud_name, time.time() - time.time()))

#     def get_batch_gen(self, split):
#         if split == 'training':
#             num_per_epoch = cfg.train_steps * cfg.batch_size
#         elif split == 'validation':
#             num_per_epoch = cfg.val_steps * cfg.val_batch_size

#         self.possibility[split] = []
#         self.min_possibility[split] = []
#         for i, tree in enumerate(self.input_colors[split]):
#             self.possibility[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]
#             self.min_possibility[split] += [float(np.min(self.possibility[split][-1]))]

#         def spatially_regular_gen():
#             for i in range(num_per_epoch):
#                 cloud_idx = int(np.argmin(self.min_possibility[split]))
#                 point_ind = np.argmin(self.possibility[split][cloud_idx])
#                 points = np.array(self.input_trees[split][cloud_idx].data, copy=False)
#                 center_point = points[point_ind, :].reshape(1, -1)
#                 noise = np.random.normal(scale=cfg.noise_init / 10, size=center_point.shape)
#                 pick_point = center_point + noise.astype(center_point.dtype)

#                 if len(points) < cfg.num_points:
#                     queried_idx = self.input_trees[split][cloud_idx].query(pick_point, k=len(points))[1][0]
#                 else:
#                     queried_idx = self.input_trees[split][cloud_idx].query(pick_point, k=cfg.num_points)[1][0]

#                 queried_idx = DP.shuffle_idx(queried_idx)
#                 queried_pc_xyz = points[queried_idx]
#                 queried_pc_xyz = queried_pc_xyz - pick_point
#                 queried_pc_colors = self.input_colors[split][cloud_idx][queried_idx]
#                 queried_pc_labels = self.input_labels[split][cloud_idx][queried_idx]

#                 dists = np.sum(np.square((points[queried_idx] - pick_point).astype(np.float32)), axis=1)
#                 delta = np.square(1 - dists / np.max(dists))
#                 self.possibility[split][cloud_idx][queried_idx] += delta
#                 self.min_possibility[split][cloud_idx] = float(np.min(self.possibility[split][cloud_idx]))

#                 if len(points) < cfg.num_points:
#                     queried_pc_xyz, queried_pc_colors, queried_idx, queried_pc_labels = \
#                         DP.data_aug(queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_idx, cfg.num_points)

#                 yield (queried_pc_xyz.astype(np.float32),
#                        queried_pc_colors.astype(np.float32),
#                        queried_pc_labels,
#                        queried_idx.astype(np.int32),
#                        np.array([cloud_idx], dtype=np.int32))

#         gen_func = spatially_regular_gen
#         gen_types = (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32)
#         gen_shapes = ([None, 3], [None, 3], [None], [None], [None])
#         return gen_func, gen_types, gen_shapes

#     @staticmethod
#     def get_tf_mapping2():
#         def tf_map(batch_xyz, batch_features, batch_labels, batch_pc_idx, batch_cloud_idx):
#             batch_features = tf.concat([batch_xyz, batch_features], axis=-1)
#             input_points = []
#             input_neighbors = []
#             input_pools = []
#             input_up_samples = []

#             for i in range(cfg.num_layers):
#                 neighbour_idx = tf.py_func(DP.knn_search, [batch_xyz, batch_xyz, cfg.k_n], tf.int32)
#                 sub_points = batch_xyz[:, :tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
#                 pool_i = neighbour_idx[:, :tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
#                 up_i = tf.py_func(DP.knn_search, [sub_points, batch_xyz, 1], tf.int32)
#                 input_points.append(batch_xyz)
#                 input_neighbors.append(neighbour_idx)
#                 input_pools.append(pool_i)
#                 input_up_samples.append(up_i)
#                 batch_xyz = sub_points

#             input_list = input_points + input_neighbors + input_pools + input_up_samples
#             input_list += [batch_features, batch_labels, batch_pc_idx, batch_cloud_idx]
#             return input_list

#         return tf_map

#     def init_input_pipeline(self):
#         print('Initiating input pipelines')
#         cfg.ignored_label_inds = [self.label_to_idx[ign_label] for ign_label in self.ignored_labels]
#         gen_function, gen_types, gen_shapes = self.get_batch_gen('training')
#         gen_function_val, _, _ = self.get_batch_gen('validation')
#         self.train_data = tf.data.Dataset.from_generator(gen_function, gen_types, gen_shapes)
#         self.val_data = tf.data.Dataset.from_generator(gen_function_val, gen_types, gen_shapes)

#         self.batch_train_data = self.train_data.batch(cfg.batch_size)
#         self.batch_val_data = self.val_data.batch(cfg.val_batch_size)
#         map_func = self.get_tf_mapping2()

#         self.batch_train_data = self.batch_train_data.map(map_func=map_func)
#         self.batch_val_data = self.batch_val_data.map(map_func=map_func)

#         self.batch_train_data = self.batch_train_data.prefetch(cfg.batch_size)
#         self.batch_val_data = self.batch_val_data.prefetch(cfg.val_batch_size)

#         iter = tf.data.Iterator.from_structure(self.batch_train_data.output_types, self.batch_train_data.output_shapes)
#         self.flat_inputs = iter.get_next()
#         self.train_init_op = iter.make_initializer(self.batch_train_data)
#         self.val_init_op = iter.make_initializer(self.batch_val_data)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--gpu', type=int, default=0, help='the number of GPUs to use [default: 0]')
#     parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')
#     parser.add_argument('--mode', type=str, default='train', help='options: train, test, vis')
#     parser.add_argument('--model_path', type=str, default='None', help='pretrained model path')
#     FLAGS = parser.parse_args()

#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#     os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)
#     os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#     Mode = FLAGS.mode

#     test_area = FLAGS.test_area
#     dataset = S3DIS(test_area)
#     dataset.init_input_pipeline()

#     if Mode == 'train':
#         model = Network(dataset, cfg)
#         model.train(dataset)
#     elif Mode == 'test':
#         cfg.saving = False
#         model = Network(dataset, cfg)
#         if FLAGS.model_path is not 'None':
#             chosen_snap = FLAGS.model_path
#         else:
#             chosen_snapshot = -1
#             logs = np.sort([os.path.join('results', f) for f in os.listdir('results') if f.startswith('Log')])
#             chosen_folder = logs[-1]
#             snap_path = join(chosen_folder, 'snapshots')
#             snap_steps = [int(f[:-5].split('-')[-1]) for f in os.listdir(snap_path) if f[-5:] == '.meta']
#             chosen_step = np.sort(snap_steps)[-1]
#             chosen_snap = os.path.join(snap_path, 'snap-{:d}'.format(chosen_step))
#         tester = ModelTester(model, dataset, restore_snap=chosen_snap)
#         tester.test(model, dataset)
#     else:
#         ##################
#         # Visualize data #
#         ##################
#         with tf.Session() as sess:
#             sess.run(tf.global_variables_initializer())
#             sess.run(dataset.train_init_op)
#             sess.run(dataset.val_init_op)
#             while True:
#                 flat_inputs = sess.run(dataset.flat_inputs)
#                 pc_xyz = flat_inputs[0]
#                 sub_pc_xyz = flat_inputs[1]
#                 labels = flat_inputs[21]
#                 Plot.draw_pc_sem_ins(pc_xyz[0, :, :], labels[0, :])
#                 Plot.draw_pc_sem_ins(sub_pc_xyz[0, :, :], labels[0, 0:np.shape(sub_pc_xyz)[1]])
