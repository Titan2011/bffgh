# from os.path import join, exists, dirname, abspath, basename, splitext
# import os
# import sys
# import glob
# import pickle
# import numpy as np
# import tensorflow as tf
# from sklearn.neighbors import KDTree

# # Make helper modules importable
# BASE_DIR = dirname(abspath(__file__))
# ROOT_DIR = dirname(BASE_DIR)
# sys.path.append(BASE_DIR)
# sys.path.append(ROOT_DIR)

# from RandLANet import Network
# from tester_Semantic3D import ModelTester
# from helper_ply import read_ply
# from helper_tool import Plot
# from helper_tool import DataProcessing as DP
# from helper_tool import ConfigSemantic3D as cfg


# class Semantic3D:
#     def __init__(self):
#         self.name = 'Semantic3D'
#         self.path = join('D:', os.sep, 'RandLA-Net', 'data', 'semantic3d')

#         # Label definitions
#         self.label_to_names = {
#             0: 'unlabeled',
#             1: 'man-made terrain',
#             2: 'natural terrain',
#             3: 'high vegetation',
#             4: 'low vegetation',
#             5: 'buildings',
#             6: 'hard scape',
#             7: 'scanning artefacts',
#             8: 'cars'
#         }
#         self.num_classes = len(self.label_to_names)
#         self.label_values = np.sort(list(self.label_to_names.keys()))
#         self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
#         self.ignored_labels = np.sort([0])

#         # Directories
#         self.original_folder = join(self.path, 'original_data')
#         self.full_pc_folder  = join(self.path, 'original_ply')
#         self.sub_pc_folder   = join(self.path, 'input_{:.3f}'.format(cfg.sub_grid_size))
#         self.gt_labels_dir   = self.original_folder

#         os.makedirs(self.full_pc_folder, exist_ok=True)
#         os.makedirs(self.sub_pc_folder, exist_ok=True)

#         # Train/val/test split
#         self.all_splits     = [0,1,4,5,3,4,3,0,1,2,3,4,2,0,5]
#         self.val_split      = 1
#         self.train_files    = []
#         self.val_files      = []
#         self.test_files     = []

#         # Discover raw scans
#         for fname in os.listdir(self.original_folder):
#             if fname.lower().endswith('.txt'):
#                 name = fname[:-4]
#                 ply_sub = join(self.sub_pc_folder, name + '.ply')
#                 ply_full = join(self.full_pc_folder, name + '.ply')
#                 lbl_file = join(self.original_folder, name + '.labels')
#                 if exists(lbl_file):
#                     self.train_files.append(ply_sub)
#                 else:
#                     self.test_files.append(ply_full)

#         self.train_files = np.sort(self.train_files)
#         self.test_files  = np.sort(self.test_files)

#         # Carve out validation set
#         for i, f in enumerate(self.train_files):
#             if self.all_splits[i] == self.val_split:
#                 self.val_files.append(f)
#         self.val_files   = np.sort(self.val_files)
#         self.train_files = np.sort([f for f in self.train_files if f not in self.val_files])

#         # Containers
#         self.val_proj, self.val_labels   = [], []
#         self.test_proj, self.test_labels = [], []
#         self.possibility   = {}
#         self.min_possibility = {}
#         self.class_weight  = {}
#         self.input_trees   = {'training':[], 'validation':[], 'test':[]}
#         self.input_colors  = {'training':[], 'validation':[], 'test':[]}
#         self.input_labels  = {'training':[], 'validation':[]}

#         # ASCII test labels
#         self.ascii_files = {
#             'MarketplaceFeldkirch_Station4_rgb_intensity-reduced.ply':   'marketsquarefeldkirch4-reduced.labels',
#             'sg27_station10_rgb_intensity-reduced.ply':                'sg27_10-reduced.labels',
#             'sg28_Station2_rgb_intensity-reduced.ply':                 'sg28_2-reduced.labels',
#             'StGallenCathedral_station6_rgb_intensity-reduced.ply':    'stgallencathedral6-reduced.labels',
#             'birdfountain_station1_xyz_intensity_rgb.ply':             'birdfountain1.labels',
#             'castleblatten_station1_intensity_rgb.ply':                'castleblatten1.labels',
#             'castleblatten_station5_xyz_intensity_rgb.ply':            'castleblatten5.labels',
#             'marketplacefeldkirch_station1_intensity_rgb.ply':         'marketsquarefeldkirch1.labels',
#             'marketplacefeldkirch_station4_intensity_rgb.ply':         'marketsquarefeldkirch4.labels',
#             'marketplacefeldkirch_station7_intensity_rgb.ply':         'marketsquarefeldkirch7.labels',
#             'sg27_station10_intensity_rgb.ply':                        'sg27_10.labels',
#             'sg27_station3_intensity_rgb.ply':                         'sg27_3.labels',
#             'sg27_station6_intensity_rgb.ply':                         'sg27_6.labels',
#             'sg27_station8_intensity_rgb.ply':                         'sg27_8.labels',
#             'sg28_station2_intensity_rgb.ply':                         'sg28_2.labels',
#             'sg28_station5_xyz_intensity_rgb.ply':                     'sg28_5.labels',
#             'stgallencathedral_station1_intensity_rgb.ply':            'stgallencathedral1.labels',
#             'stgallencathedral_station3_intensity_rgb.ply':            'stgallencathedral3.labels',
#             'stgallencathedral_station6_intensity_rgb.ply':            'stgallencathedral6.labels'
#         }

#         self.load_sub_sampled_clouds(cfg.sub_grid_size)

#     def load_sub_sampled_clouds(self, sub_grid_size):
#         tree_path = join(self.path, 'input_{:.3f}'.format(sub_grid_size))
#         all_files = np.hstack((self.train_files, self.val_files, self.test_files))

#         for i, fpath in enumerate(all_files):
#             name = splitext(basename(fpath))[0]
#             print('Load_pc_{i}: {name}'.format(i=i, name=name))

#             if fpath in self.val_files:
#                 split = 'validation'
#             elif fpath in self.train_files:
#                 split = 'training'
#             else:
#                 split = 'test'

#             data = read_ply(join(tree_path, name + '.ply'))
#             colors = np.vstack((data['red'], data['green'], data['blue'])).T
#             labels = None if split == 'test' else data['class']

#             with open(join(tree_path, name + '_KDTree.pkl'), 'rb') as f:
#                 tree = pickle.load(f)

#             self.input_trees[split].append(tree)
#             self.input_colors[split].append(colors)
#             if split in ['training', 'validation']:
#                 self.input_labels[split].append(labels)

#         print('\nPreparing reprojection indices for validation and test')
#         for fpath in all_files:
#             name = splitext(basename(fpath))[0]
#             proj_file = join(tree_path, name + '_proj.pkl')
#             with open(proj_file, 'rb') as f:
#                 proj_idx, lbls = pickle.load(f)
#             if fpath in self.val_files:
#                 self.val_proj.append(proj_idx)
#                 self.val_labels.append(lbls)
#             if fpath in self.test_files:
#                 self.test_proj.append(proj_idx)
#                 self.test_labels.append(lbls)
#         print('finished')

#     def get_batch_gen(self, split):
#         # Explicit split-to-size mapping
#         if split == 'training':
#             num_per_epoch = cfg.train_steps * cfg.batch_size
#         elif split in ('validation', 'test'):
#             num_per_epoch = cfg.val_steps * cfg.val_batch_size
#         else:
#             raise ValueError("Unsupported split '{}'".format(split))

#         self.possibility[split]     = []
#         self.min_possibility[split] = []
#         self.class_weight[split]    = []

#         for tree in self.input_trees[split]:
#             p = np.random.rand(tree.data.shape[0]) * 1e-3
#             self.possibility[split].append(p)
#             self.min_possibility[split].append(float(np.min(p)))

#         if split != 'test':
#             counts = np.unique(np.hstack(self.input_labels[split]), return_counts=True)[1]
#             cw = counts.astype(np.float32) / np.sum(counts)
#             self.class_weight[split].append(cw)

#         def spatially_regular_gen():
#             for _ in range(num_per_epoch):
#                 c_idx = int(np.argmin(self.min_possibility[split]))
#                 tree  = self.input_trees[split][c_idx]

#                 # Convert to ndarray for reshape()
#                 points = np.asarray(tree.data)

#                 pt_idx = int(np.argmin(self.possibility[split][c_idx]))
#                 center = points[pt_idx].reshape(1, -1)

#                 noise = np.random.normal(scale=cfg.noise_init/10, size=center.shape)
#                 pick  = center + noise.astype(center.dtype)

#                 qidx = tree.query(pick, k=cfg.num_points)[1][0]
#                 qidx = DP.shuffle_idx(qidx)

#                 xyz = points[qidx]
#                 xyz[:, :2] -= pick[:, :2]
#                 col = self.input_colors[split][c_idx][qidx]

#                 if split == 'test':
#                     lbl = np.zeros(xyz.shape[0], dtype=np.int32)
#                     w   = 1.0
#                 else:
#                     raw_lbl = self.input_labels[split][c_idx][qidx]
#                     lbl     = np.array([self.label_to_idx[l] for l in raw_lbl])
#                     w       = np.array([self.class_weight[split][0][n] for n in lbl])

#                 d     = np.sum((points[qidx] - pick)**2, axis=1)
#                 delta = (1 - d/np.max(d))**2 * w
#                 self.possibility[split][c_idx][qidx] += delta
#                 self.min_possibility[split][c_idx]   = float(np.min(self.possibility[split][c_idx]))

#                 yield (xyz.astype(np.float32),
#                        col.astype(np.float32),
#                        lbl.astype(np.int32),
#                        qidx.astype(np.int32),
#                        np.array([c_idx], dtype=np.int32))

#         types  = (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32)
#         shapes = ([None,3], [None,3], [None], [None], [None])
#         return spatially_regular_gen, types, shapes

#     def get_tf_mapping(self):
#         def tf_map(batch_xyz, batch_feat, batch_lbl, batch_pc_idx, batch_cloud_idx):
#             batch_feat = tf.map_fn(self.tf_augment_input,
#                                    [batch_xyz, batch_feat],
#                                    dtype=tf.float32)
#             pts_list, neigh_list, pool_list, up_list = [], [], [], []
#             cur_xyz = batch_xyz
#             for i in range(cfg.num_layers):
#                 neigh = tf.py_func(DP.knn_search,
#                                    [cur_xyz, cur_xyz, cfg.k_n],
#                                    tf.int32)
#                 sub  = cur_xyz[:, :tf.shape(cur_xyz)[1]//cfg.sub_sampling_ratio[i], :]
#                 pool = neigh[:, :tf.shape(cur_xyz)[1]//cfg.sub_sampling_ratio[i], :]
#                 up   = tf.py_func(DP.knn_search, [sub, cur_xyz, 1], tf.int32)
#                 pts_list.append(cur_xyz)
#                 neigh_list.append(neigh)
#                 pool_list.append(pool)
#                 up_list.append(up)
#                 cur_xyz = sub

#             inputs = pts_list + neigh_list + pool_list + up_list
#             inputs += [batch_feat, batch_lbl, batch_pc_idx, batch_cloud_idx]
#             return inputs
#         return tf_map

#     @staticmethod
#     def tf_augment_input(inputs):
#         xyz, feats = inputs
#         theta = tf.random_uniform((1,), 0, 2*np.pi)
#         c, s  = tf.cos(theta), tf.sin(theta)
#         cs0, cs1 = tf.zeros_like(c), tf.ones_like(c)
#         R = tf.reshape(tf.stack([c,-s,cs0, s, c,cs0, cs0,cs0,cs1]), (3,3))
#         xyz_t = tf.reshape(tf.matmul(xyz, R), [-1,3])

#         if cfg.augment_scale_anisotropic:
#             scale = tf.random_uniform((1,3), cfg.augment_scale_min, cfg.augment_scale_max)
#         else:
#             scale = tf.random_uniform((1,1), cfg.augment_scale_min, cfg.augment_scale_max)

#         syms = [
#             (tf.round(tf.random_uniform((1,1)))*2-1) if cfg.augment_symmetries[i]
#             else tf.ones((1,1), dtype=tf.float32)
#             for i in range(3)
#         ]
#         scale = tf.concat(syms,1) * scale
#         xyz_t = xyz_t * tf.tile(scale, [tf.shape(xyz_t)[0],1])
#         xyz_t = xyz_t + tf.random_normal(tf.shape(xyz_t), stddev=cfg.augment_noise)

#         rgb = feats[:, :3]
#         return tf.concat([xyz_t, rgb], axis=-1)

#     def init_input_pipeline(self):
#         print('Initiating input pipelines')
#         cfg.ignored_label_inds = [self.label_to_idx[i] for i in self.ignored_labels]

#         gen_tr, t_tr, s_tr = self.get_batch_gen('training')
#         gen_va, t_va, s_va = self.get_batch_gen('validation')
#         gen_te, t_te, s_te = self.get_batch_gen('test')

#         ds_tr  = tf.data.Dataset.from_generator(gen_tr, t_tr, s_tr).batch(cfg.batch_size)
#         ds_val = tf.data.Dataset.from_generator(gen_va, t_va, s_va).batch(cfg.val_batch_size)
#         ds_te  = tf.data.Dataset.from_generator(gen_te, t_te, s_te).batch(cfg.val_batch_size)

#         mapper = self.get_tf_mapping()
#         self.batch_train_data = ds_tr.map(mapper).prefetch(cfg.batch_size)
#         self.batch_val_data   = ds_val.map(mapper).prefetch(cfg.val_batch_size)
#         self.batch_test_data  = ds_te.map(mapper).prefetch(cfg.val_batch_size)

#         it = tf.data.Iterator.from_structure(self.batch_train_data.output_types,
#                                               self.batch_train_data.output_shapes)
#         self.flat_inputs   = it.get_next()
#         self.train_init_op = it.make_initializer(self.batch_train_data)
#         self.val_init_op   = it.make_initializer(self.batch_val_data)
#         self.test_init_op  = it.make_initializer(self.batch_test_data)


# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--gpu', type=int, default=0, help='GPU to use')
#     parser.add_argument('--mode', type=str, default='train', help='train | test | vis')
#     parser.add_argument('--model_path', type=str, default='None', help='pretrained model path')
#     FLAGS = parser.parse_args()

#     os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
#     os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
#     os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

#     dataset = Semantic3D()
#     dataset.init_input_pipeline()

#     if FLAGS.mode == 'train':
#         model = Network(dataset, cfg)
#         model.train(dataset)

#     elif FLAGS.mode == 'test':
#         cfg.saving = False
#         model = Network(dataset, cfg)
#         if FLAGS.model_path != 'None':
#             snap = FLAGS.model_path
#         else:
#             logs = np.sort([join('results', d) for d in os.listdir('results') if d.startswith('Log')])
#             last = logs[-1]
#             snaps_dir = join(last, 'snapshots')
#             steps = [
#                 int(splitext(f)[0].split('-')[-1])
#                 for f in os.listdir(snaps_dir)
#                 if f.endswith('.meta')
#             ]
#             last_step = max(steps)
#             snap = join(snaps_dir, 'snap-{}.meta'.format(last_step))
#         tester = ModelTester(model, dataset, restore_snap=snap)
#         tester.test(model, dataset)

#     else:  # visualization
#         with tf.Session() as sess:
#             sess.run(tf.global_variables_initializer())
#             sess.run(dataset.train_init_op)
#             while True:
#                 flat_inputs = sess.run(dataset.flat_inputs)
#                 pc_xyz = flat_inputs[0]
#                 sub_pc_xyz = flat_inputs[1]
#                 labels = flat_inputs[21]
#                 Plot.draw_pc_sem_ins(pc_xyz[0, :, :], labels[0, :])
#                 Plot.draw_pc_sem_ins(sub_pc_xyz[0, :, :], labels[0, 0:np.shape(sub_pc_xyz)[1]])



import os
import sys
import pickle
import numpy as np
import tensorflow as tf
from os.path import dirname, abspath, join, splitext, basename, exists

# Make helper modules importable
BASE_DIR = dirname(abspath(__file__))
ROOT_DIR = dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)

# Import our new architecture
from RandLANet import Network
from tester_Semantic3D import ModelTester
from helper_ply import read_ply
from helper_tool import Plot
from helper_tool import DataProcessing as DP
from helper_tool import ConfigSemantic3D as cfg
from helper_tool import Plot
import time

class Semantic3D:
    def __init__(self):
        self.name = 'Semantic3D'
        self.path = join('D:', os.sep, 'RandLA-Net', 'data', 'semantic3d')

        # Label definitions
        self.label_to_names = {
            0: 'unlabeled',
            1: 'man-made terrain',
            2: 'natural terrain',
            3: 'high vegetation',
            4: 'low vegetation',
            5: 'buildings',
            6: 'hard scape',
            7: 'scanning artefacts',
            8: 'cars'
        }
        self.num_classes     = len(self.label_to_names)
        self.label_values    = np.sort(list(self.label_to_names.keys()))
        self.label_to_idx    = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels  = np.sort([0])

        # Directories
        self.original_folder = join(self.path, 'original_data')
        self.full_pc_folder  = join(self.path, 'original_ply')
        self.sub_pc_folder   = join(self.path, 'input_{:.3f}'.format(cfg.sub_grid_size))
        self.gt_labels_dir   = self.original_folder

        os.makedirs(self.full_pc_folder, exist_ok=True)
        os.makedirs(self.sub_pc_folder, exist_ok=True)

        # Train/val/test split
        self.all_splits  = [0, 1, 4, 5, 3, 4, 3, 0, 1, 2, 3, 4, 2, 0, 5]
        self.val_split   = 1
        self.train_files = []
        self.val_files   = []
        self.test_files  = []

        # Discover raw scans
        for fname in os.listdir(self.original_folder):
            if fname.lower().endswith('.txt'):
                name     = fname[:-4]
                ply_sub  = join(self.sub_pc_folder, name + '.ply')
                ply_full = join(self.full_pc_folder, name + '.ply')
                lbl_file = join(self.original_folder, name + '.labels')
                if exists(lbl_file):
                    self.train_files.append(ply_sub)
                else:
                    self.test_files.append(ply_full)

        self.train_files = np.sort(self.train_files)
        self.test_files  = np.sort(self.test_files)

        # Carve out validation set
        for i, f in enumerate(self.train_files):
            if self.all_splits[i] == self.val_split:
                self.val_files.append(f)
        self.val_files   = np.sort(self.val_files)
        self.train_files = np.sort([f for f in self.train_files if f not in self.val_files])

        # Containers
        self.val_proj, self.val_labels   = [], []
        self.test_proj, self.test_labels = [], []
        self.possibility      = {}
        self.min_possibility  = {}
        self.class_weight     = {}
        self.input_trees      = {'training': [], 'validation': [], 'test': []}
        self.input_colors     = {'training': [], 'validation': [], 'test': []}
        self.input_labels     = {'training': [], 'validation': []}

        # ASCII test labels
        self.ascii_files = {
            'MarketplaceFeldkirch_Station4_rgb_intensity-reduced.ply':   'marketsquarefeldkirch4-reduced.labels',
            'sg27_station10_rgb_intensity-reduced.ply':                'sg27_10-reduced.labels',
            'sg28_Station2_rgb_intensity-reduced.ply':                 'sg28_2-reduced.labels',
            'StGallenCathedral_station6_rgb_intensity-reduced.ply':    'stgallencathedral6-reduced.labels',
            'birdfountain_station1_xyz_intensity_rgb.ply':             'birdfountain1.labels',
            'castleblatten_station1_intensity_rgb.ply':                'castleblatten1.labels',
            'castleblatten_station5_xyz_intensity_rgb.ply':            'castleblatten5.labels',
            'marketplacefeldkirch_station1_intensity_rgb.ply':         'marketsquarefeldkirch1.labels',
            'marketplacefeldkirch_station4_intensity_rgb.ply':         'marketsquarefeldkirch4.labels',
            'marketplacefeldkirch_station7_intensity_rgb.ply':         'marketsquarefeldkirch7.labels',
            'sg27_station10_intensity_rgb.ply':                        'sg27_10.labels',
            'sg27_station3_intensity_rgb.ply':                         'sg27_3.labels',
            'sg27_station6_intensity_rgb.ply':                         'sg27_6.labels',
            'sg27_station8_intensity_rgb.ply':                         'sg27_8.labels',
            'sg28_station2_intensity_rgb.ply':                         'sg28_2.labels',
            'sg28_station5_xyz_intensity_rgb.ply':                     'sg28_5.labels',
            'stgallencathedral_station1_intensity_rgb.ply':            'stgallencathedral1.labels',
            'stgallencathedral_station3_intensity_rgb.ply':            'stgallencathedral3.labels',
            'stgallencathedral_station6_intensity_rgb.ply':            'stgallencathedral6.labels'
        }

        self.load_sub_sampled_clouds(cfg.sub_grid_size)

    def load_sub_sampled_clouds(self, sub_grid_size):
        tree_path = join(self.path, 'input_{:.3f}'.format(sub_grid_size))
        all_files = np.hstack((self.train_files, self.val_files, self.test_files))

        for i, fpath in enumerate(all_files):
            name = splitext(basename(fpath))[0]
            print('Load_pc_{}: {}'.format(i, name))

            if fpath in self.val_files:
                split = 'validation'
            elif fpath in self.train_files:
                split = 'training'
            else:
                split = 'test'

            data = read_ply(join(tree_path, name + '.ply'))
            # We only need RGB + XYZ coordinates
            colors = np.vstack((data['red'], data['green'], data['blue'])).T  # [N,3]
            labels = None if split == 'test' else data['class']                 # [N,]

            with open(join(tree_path, '{}_KDTree.pkl'.format(name)), 'rb') as f:
                tree = pickle.load(f)

            self.input_trees[split].append(tree)
            self.input_colors[split].append(colors)
            if split in ['training', 'validation']:
                self.input_labels[split].append(labels)

        print('\nPreparing reprojection indices for validation and test')
        for fpath in all_files:
            name = splitext(basename(fpath))[0]
            with open(join(tree_path, '{}_proj.pkl'.format(name)), 'rb') as f:
                proj_idx, lbls = pickle.load(f)
            if fpath in self.val_files:
                self.val_proj.append(proj_idx)
                self.val_labels.append(lbls)
            if fpath in self.test_files:
                self.test_proj.append(proj_idx)
                self.test_labels.append(lbls)
        print('finished')

    def get_batch_gen(self, split):
        if split == 'training':
            num_per_epoch = cfg.train_steps * cfg.batch_size
        elif split in ('validation', 'test'):
            num_per_epoch = cfg.val_steps * cfg.val_batch_size
        else:
            raise ValueError("Unsupported split '{}'".format(split))

        self.possibility[split]     = []
        self.min_possibility[split] = []
        self.class_weight[split]    = []

        for tree in self.input_trees[split]:
            p = np.random.rand(tree.data.shape[0]) * 1e-3
            self.possibility[split].append(p)
            self.min_possibility[split].append(float(np.min(p)))

        if split != 'test':
            counts = np.unique(np.hstack(self.input_labels[split]), return_counts=True)[1]
            cw = counts.astype(np.float32) / np.sum(counts)
            self.class_weight[split].append(cw)

        def spatially_regular_gen():
            for _ in range(num_per_epoch):
                c_idx = int(np.argmin(self.min_possibility[split]))
                tree  = self.input_trees[split][c_idx]

                points = np.asarray(tree.data)
                pt_idx = int(np.argmin(self.possibility[split][c_idx]))
                center = points[pt_idx].reshape(1, -1)

                noise = np.random.normal(scale=cfg.noise_init / 10, size=center.shape)
                pick  = center + noise.astype(center.dtype)

                qidx = tree.query(pick, k=cfg.num_points)[1][0]
                qidx = DP.shuffle_idx(qidx)

                xyz = points[qidx]  # [N, 3]
                xyz[:, :2] -= pick[:, :2]
                col = self.input_colors[split][c_idx][qidx]  # [N, 3]

                # Combine XYZ + RGB into 6-channel features
                feats = np.concatenate([xyz, col], axis=1).astype(np.float32)  # [N, 6]

                if split == 'test':
                    lbl = np.zeros(xyz.shape[0], dtype=np.int32)
                    w   = 1.0
                else:
                    raw_lbl = self.input_labels[split][c_idx][qidx]
                    lbl     = np.array([self.label_to_idx[l] for l in raw_lbl])
                    w       = np.array([self.class_weight[split][0][n] for n in lbl])

                d     = np.sum((points[qidx] - pick) ** 2, axis=1)
                delta = (1 - d / np.max(d)) ** 2 * w
                self.possibility[split][c_idx][qidx] += delta
                self.min_possibility[split][c_idx]   = float(np.min(self.possibility[split][c_idx]))

                yield (
                    xyz.astype(np.float32),
                    feats,                    # [N, 6] = XYZ+RGB
                    lbl.astype(np.int32),
                    qidx.astype(np.int32),
                    np.array([c_idx], dtype=np.int32)
                )

        types  = (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32)
        shapes = ([None, 3], [None, 6], [None], [None], [None])
        return spatially_regular_gen, types, shapes

    def get_tf_mapping(self):
        """
        Returns a mapping function that takes (batch_xyz, batch_feat, batch_lbl,
        batch_pc_idx, batch_cloud_idx) and returns a list of 29 tensors:
        [xyz_0, neigh_idx_0, sub_idx_0, pool_idx_0, up_idx_0,
         xyz_1, neigh_idx_1, sub_idx_1, pool_idx_1, up_idx_1,
         ...,  (for all 5 layers),
         batch_feat, batch_lbl, batch_pc_idx, batch_cloud_idx]
        """

        def tf_map(batch_xyz, batch_feat, batch_lbl, batch_pc_idx, batch_cloud_idx):
            # 1) Apply data augmentation (unchanged)
            batch_feat = tf.map_fn(self.tf_augment_input, [batch_xyz, batch_feat], dtype=tf.float32)

            pts_list   = []  # [xyz_0, xyz_1, ..., xyz_4]
            neigh_list = []  # [neigh_idx_0, ..., neigh_idx_4]
            sub_list   = []  # [sub_idx_0, ..., sub_idx_4]
            pool_list  = []  # [pool_idx_0, ..., pool_idx_4]
            up_list    = []  # [up_idx_0, ..., up_idx_4]

            cur_xyz = batch_xyz  # [B, N, 3]  for layer 0

            for i in range(cfg.num_layers):  # L = 5
                # --- a) KNN search on current xyz_i ---
                neigh = tf.py_func(DP.knn_search, [cur_xyz, cur_xyz, cfg.k_n], tf.int32)
                # neigh: [B, N_i, k_n]

                # --- b) Subsample indices (sub_idx) ---
                N_i = tf.shape(cur_xyz)[1]  # number of points in this layer
                sub_N = tf.cast(N_i // cfg.sub_sampling_ratio[i], tf.int32)

                # Create a [sub_N] range for each batch
                idx_range = tf.range(N_i)  # [N_i]
                # The first sub_N indices (0,1,2,...,sub_N-1)
                sub_idx_single = idx_range[:sub_N]  # [sub_N]
                # Tile to shape [B, sub_N]
                sub_idx = tf.tile(tf.reshape(sub_idx_single, [1, -1]), [tf.shape(cur_xyz)[0], 1])  # [B, sub_N]

                # --- c) pool_idx: take first sub_N neighbors per point ---
                pool = neigh[:, :sub_N, :]  # [B, sub_N, k_n]

                # --- d) up_idx: 1-NN from sub_xyz back to original xyz_i ---
                sub_xyz = cur_xyz[:, :sub_N, :]  # [B, sub_N, 3]
                up = tf.py_func(DP.knn_search, [sub_xyz, cur_xyz, 1], tf.int32)  # [B, sub_N, 1]

                # Append all five for layer i
                pts_list.append(cur_xyz)   # xyz_i
                neigh_list.append(neigh)   # neigh_idx_i
                sub_list.append(sub_idx)   # sub_idx_i
                pool_list.append(pool)     # pool_idx_i
                up_list.append(up)         # up_idx_i

                # Move to next layer’s xyz (subsampled)
                cur_xyz = sub_xyz

            # After all layers, append the four static tensors:
            # batch_feat: [B, N, num_features]
            # batch_lbl:  [B, N]
            # batch_pc_idx: [B]  (indices into dataset)
            # batch_cloud_idx: [B]
            all_inputs = pts_list + neigh_list + sub_list + pool_list + up_list
            all_inputs += [batch_feat, batch_lbl, batch_pc_idx, batch_cloud_idx]

            return all_inputs  # total length = 5*L + 4 = 29

        return tf_map

    @staticmethod
    def tf_augment_input(inputs):
        xyz, feats = inputs  # feats not used; we use feats = [XYZ,RGB] directly
        return feats  # identity (no extra augmentation), since we built feats already

    def init_input_pipeline(self):
        print('Initiating input pipelines')
        cfg.ignored_label_inds = [self.label_to_idx[i] for i in self.ignored_labels]

        gen_tr, t_tr, s_tr = self.get_batch_gen('training')
        gen_va, t_va, s_va = self.get_batch_gen('validation')
        gen_te, t_te, s_te = self.get_batch_gen('test')

        ds_tr  = tf.data.Dataset.from_generator(gen_tr, t_tr, s_tr).batch(cfg.batch_size)
        ds_val = tf.data.Dataset.from_generator(gen_va, t_va, s_va).batch(cfg.val_batch_size)
        ds_te  = tf.data.Dataset.from_generator(gen_te, t_te, s_te).batch(cfg.val_batch_size)

        mapper = self.get_tf_mapping()
        self.batch_train_data = ds_tr.map(mapper).prefetch(cfg.batch_size)
        self.batch_val_data   = ds_val.map(mapper).prefetch(cfg.val_batch_size)
        self.batch_test_data  = ds_te.map(mapper).prefetch(cfg.val_batch_size)

        it = tf.data.Iterator.from_structure(
            self.batch_train_data.output_types,
            self.batch_train_data.output_shapes
        )
        self.flat_inputs   = it.get_next()
        self.train_init_op = it.make_initializer(self.batch_train_data)
        self.val_init_op   = it.make_initializer(self.batch_val_data)
        self.test_init_op  = it.make_initializer(self.batch_test_data)


if __name__ == '__main__':
    import argparse

    def log_out(msg, log_file):
        print(msg)
        log_file.write(msg + '\n')
        log_file.flush()

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',        type=int,   default=0,   help='GPU to use')
    parser.add_argument('--mode',       type=str,   default='train', help='train | test | vis')
    parser.add_argument('--model_path', type=str,   default='None',  help='pretrained model path')
    FLAGS = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    # ─── Ensure saving_path is set early ───
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    cfg.saving_path = join('results', 'Sem3D_Run_' + timestamp)
    if not exists(cfg.saving_path):
        os.makedirs(cfg.saving_path)

    dataset = Semantic3D()
    dataset.init_input_pipeline()

    print("\n" + "="*40)
    print("--- RUNNING INPUT SHAPE VALIDATION ---")
    print("="*40)
    with tf.Session() as sess:
        # Initialize the training data pipeline
        sess.run(dataset.train_init_op)

        # Fetch one batch of data from the pipeline
        fetched_inputs = sess.run(dataset.flat_inputs)
        
        # Get the number of layers from your config
        L = cfg.num_layers

        # According to your Network class, it expects features at index 4*L
        # According to your get_tf_mapping function, the features are at index 5*L
        
        xyz_tensor_from_pipeline = fetched_inputs[0]
        
        # This is what your Network class *thinks* is the features tensor
        supposed_features_tensor = fetched_inputs[4*L] 
        
        # This is what your pipeline *actually* prepared as the features tensor
        actual_features_tensor = fetched_inputs[5*L]

        print("Number of layers (L) = {}\n".format(L))
        print("Shape of initial XYZ data (index 0): {}".format(xyz_tensor_from_pipeline))
        print("Data at index 4*L = {} (what Network expects for features): {}".format(4*L, supposed_features_tensor.shape))
        print("Data at index 5*L = {} (what pipeline actually provides as features): {}".format(5*L, actual_features_tensor.shape))
        print("\nNOTICE: The shape at index 4*L does not look like a features tensor.")
        print("This mismatch is the source of the error.")
        print("="*40 + "\n")

    if FLAGS.mode == 'train':
        model = Network(dataset, cfg)
        model.train(dataset)

    elif FLAGS.mode == 'test':
        cfg.saving = False
        model = Network(dataset, cfg)
        if FLAGS.model_path != 'None':
            snap = FLAGS.model_path
        else:
            logs = np.sort([join('results', d) for d in os.listdir('results') if d.startswith('Log')])
            last = logs[-1]
            snaps_dir = join(last, 'snapshots')
            steps = [
                int(splitext(f)[0].split('-')[-1])
                for f in os.listdir(snaps_dir)
                if f.endswith('.meta')
            ]
            last_step = max(steps)
            snap = join(snaps_dir, 'snap-{}.meta'.format(last_step))
        tester = ModelTester(model, dataset, restore_snap=snap)
        tester.test(model, dataset)

    else:  # visualization
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(dataset.train_init_op)
            while True:
                flat_inputs = sess.run(dataset.flat_inputs)
                pc_xyz     = flat_inputs[0]
                sub_pc_xyz = flat_inputs[1]
                labels     = flat_inputs[21]
                Plot.draw_pc_sem_ins(pc_xyz[0, :, :], labels[0, :])
                Plot.draw_pc_sem_ins(sub_pc_xyz[0, :, :], labels[0, 0:np.shape(sub_pc_xyz)[1]])
