# from os import makedirs
# from os.path import exists, join, basename, dirname
# from helper_ply import read_ply, write_ply
# import tensorflow as tf
# import numpy as np
# import time


# def log_string(out_str, log_out):
#     log_out.write(out_str + '\n')
#     log_out.flush()
#     print(out_str)


# class ModelTester(object):
#     def __init__(self, model, dataset, restore_snap=None):
#         # Tensorflow Saver definition
#         my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
#         self.saver = tf.train.Saver(my_vars, max_to_keep=100)

#         # Create a session for running Ops on the Graph.
#         c_proto = tf.ConfigProto()
#         c_proto.gpu_options.allow_growth = True
#         self.sess = tf.Session(config=c_proto)
#         self.sess.run(tf.global_variables_initializer())

#         if restore_snap is not None:
#             # If a .meta file was passed in, strip its extension to get the checkpoint prefix
#             if restore_snap.lower().endswith('.meta'):
#                 restore_snap = restore_snap[:-5]
#             self.saver.restore(self.sess, restore_snap)
#             print("Model restored from " + restore_snap)

#         # Add a softmax operation for predictions
#         self.prob_logits = tf.nn.softmax(model.logits)
#         self.test_probs = [
#             np.zeros((l.data.shape[0], model.config.num_classes), dtype=np.float16)
#             for l in dataset.input_trees['test']
#         ]

#         self.log_out = open('log_test_' + dataset.name + '.txt', 'a')

#     def test(self, model, dataset, num_votes=100):
#         test_smooth = 0.98

#         # Initialise iterator with test data
#         self.sess.run(dataset.test_init_op)

#         # Test saving path
#         time_tag    = time.strftime('Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
#         saving_path = join('results', time_tag)
#         test_path   = join('test',    time_tag)
#         if not exists(test_path):
#             makedirs(test_path)
#         pred_dir = join(test_path, 'predictions')
#         prob_dir = join(test_path, 'probs')
#         if not exists(pred_dir):
#             makedirs(pred_dir)
#         if not exists(prob_dir):
#             makedirs(prob_dir)

#         step_id, epoch_id = 0, 0
#         last_min = -0.5

#         while last_min < num_votes:
#             try:
#                 ops = (
#                     self.prob_logits,
#                     model.labels,
#                     model.inputs['input_inds'],
#                     model.inputs['cloud_inds'],
#                 )
#                 stacked_probs, _, point_idx, cloud_idx = self.sess.run(
#                     ops, {model.is_training: False}
#                 )

#                 # reshape by known batch dims
#                 bsize = model.config.val_batch_size
#                 npts  = model.config.num_points
#                 ncls  = model.config.num_classes
#                 stacked_probs = np.reshape(stacked_probs, (bsize, npts, ncls))

#                 for j in range(bsize):
#                     probs = stacked_probs[j]
#                     inds  = point_idx[j]
#                     c_i   = cloud_idx[j][0]
#                     self.test_probs[c_i][inds] = (
#                         test_smooth * self.test_probs[c_i][inds]
#                         + (1 - test_smooth) * probs
#                     )

#                 step_id += 1
#                 log_string(
#                     'Epoch {:3d}, step {:3d}. min possibility = {:.1f}'.format(
#                         epoch_id, step_id,
#                         np.min(dataset.min_possibility['test'])
#                     ),
#                     self.log_out
#                 )

#             except tf.errors.OutOfRangeError:
#                 new_min = np.min(dataset.min_possibility['test'])
#                 log_string(
#                     'Epoch {:3d}, end. Min possibility = {:.1f}'.format(
#                         epoch_id, new_min
#                     ),
#                     self.log_out
#                 )

#                 if last_min + 4 < new_min:
#                     print('Saving clouds')
#                     last_min = new_min

#                     print('\nReproject Vote #{}'.format(int(np.floor(new_min))))
#                     t1 = time.time()

#                     for i_test, file_path in enumerate(dataset.test_files):
#                         # Load original points
#                         pts = self.load_evaluation_points(file_path).astype(np.float16)

#                         # Reproject probabilities
#                         proj_idx = dataset.test_proj[i_test]
#                         probs    = self.test_probs[i_test][proj_idx]

#                         # Insert zeros for ignored labels
#                         probs2 = probs
#                         for li, lv in enumerate(dataset.label_values):
#                             if lv in dataset.ignored_labels:
#                                 probs2 = np.insert(probs2, li, 0, axis=1)

#                         # Predicted labels
#                         preds = dataset.label_values[np.argmax(probs2, axis=1)].astype(np.uint8)

#                         # Save ASCII preds
#                         cloud_name = basename(file_path)
#                         ascii_name = dataset.ascii_files.get(cloud_name,
#                                                             cloud_name + '.labels')
#                         out_path   = join(pred_dir, ascii_name)
#                         np.savetxt(out_path, preds, fmt='%d')
#                         log_string(out_path + ' has been saved', self.log_out)

#                     t2 = time.time()
#                     print('Done in {:.1f} s\n'.format(t2 - t1))
#                     self.sess.close()
#                     return

#                 # restart for next epoch
#                 self.sess.run(dataset.test_init_op)
#                 epoch_id += 1
#                 step_id = 0
#                 continue

#     @staticmethod
#     def load_evaluation_points(file_path):
#         data = read_ply(file_path)
#         return np.vstack((data['x'], data['y'], data['z'])).T



from os import makedirs
from os.path import exists, join, basename
from helper_ply import read_ply, write_ply
from sklearn.metrics import confusion_matrix
from helper_tool import DataProcessing as DP
import tensorflow as tf
import numpy as np
import time

def log_string(out_str, log_out):
    log_out.write(out_str + '\n')
    log_out.flush()
    print(out_str)

class ModelTester(object):
    def __init__(self, model, dataset, restore_snap=None):
        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.saver = tf.train.Saver(my_vars, max_to_keep=100)

        c_proto = tf.ConfigProto()
        c_proto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=c_proto)
        self.sess.run(tf.global_variables_initializer())

        if restore_snap is not None:
            if restore_snap.lower().endswith('.meta'):
                restore_snap = restore_snap[:-5]
            self.saver.restore(self.sess, restore_snap)
            print("Model restored from {}".format(restore_snap))

        self.prob_logits = tf.nn.softmax(model.logits)
        self.test_probs = [
            np.zeros((l.data.shape[0], model.config.num_classes), dtype=np.float16)
            for l in dataset.input_trees['test']
        ]

        self.log_out = open('log_test_{}.txt'.format(dataset.name), 'a')

    def test(self, model, dataset, num_votes=100):
        test_smooth = 0.98
        self.sess.run(dataset.test_init_op)

        time_tag = time.strftime('Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
        test_path = join('test', time_tag)
        pred_dir = join(test_path, 'predictions')
        prob_dir = join(test_path, 'probs')
        for path in [test_path, pred_dir, prob_dir]:
            makedirs(path, exist_ok=True)

        step_id, epoch_id = 0, 0
        last_min = -0.5

        while last_min < num_votes:
            try:
                ops = (
                    self.prob_logits,
                    model.labels,
                    model.inputs['input_inds'],
                    model.inputs['cloud_inds'],
                )
                stacked_probs, _, point_idx, cloud_idx = self.sess.run(
                    ops, {model.is_training: False}
                )

                bsize = model.config.val_batch_size
                npts = model.config.num_points
                ncls = model.config.num_classes
                stacked_probs = np.reshape(stacked_probs, (bsize, npts, ncls))

                for j in range(bsize):
                    probs = stacked_probs[j]
                    inds = point_idx[j]
                    c_i = cloud_idx[j][0]
                    self.test_probs[c_i][inds] = (
                        test_smooth * self.test_probs[c_i][inds]
                        + (1 - test_smooth) * probs
                    )

                step_id += 1
                log_string(
                    'Epoch {:3d}, step {:3d}. min possibility = {:.1f}'.format(
                        epoch_id, step_id,
                        np.min(dataset.min_possibility['test'])
                    ),
                    self.log_out
                )

            except tf.errors.OutOfRangeError:
                new_min = np.min(dataset.min_possibility['test'])
                log_string(
                    'Epoch {:3d}, end. Min possibility = {:.1f}'.format(
                        epoch_id, new_min
                    ),
                    self.log_out
                )

                if last_min + 4 < new_min:
                    last_min = new_min
                    log_string(
                        '\nReproject Vote #{}'.format(int(np.floor(new_min))),
                        self.log_out
                    )

                    confusion_list = []

                    for i_test, file_path in enumerate(dataset.test_files):
                        # Load original points
                        pts = self.load_evaluation_points(file_path).astype(np.float16)

                        # Reproject probabilities
                        proj_idx = dataset.test_proj[i_test]
                        probs = self.test_probs[i_test][proj_idx]

                        # Insert zeros for ignored labels
                        probs2 = probs
                        for li, lv in enumerate(dataset.label_values):
                            if lv in dataset.ignored_labels:
                                probs2 = np.insert(probs2, li, 0, axis=1)

                        # Predictions
                        preds = dataset.label_values[np.argmax(probs2, axis=1)].astype(np.uint8)

                        # Save ASCII predictions
                        cloud_name = basename(file_path)
                        ascii_name = dataset.ascii_files.get(cloud_name,
                                                            cloud_name + '.labels')
                        out_path = join(pred_dir, ascii_name)
                        np.savetxt(out_path, preds, fmt='%d')

                        # Load ground truth labels (for IoU)
                        gt_path = dataset.label_files.get(cloud_name)
                        if gt_path:
                            gt_labels = np.loadtxt(gt_path, dtype=np.uint8)
                            confusion_list.append(
                                confusion_matrix(gt_labels, preds, dataset.label_values)
                            )

                    # Compute IoUs if ground truths are available
                    if confusion_list:
                        C = np.sum(np.stack(confusion_list), axis=0).astype(np.float32)
                        IoUs = DP.IoU_from_confusions(C)
                        m_IoU = np.mean(IoUs)
                        s = '{:5.2f} | '.format(100 * m_IoU)
                        for IoU in IoUs:
                            s += '{:5.2f} '.format(100 * IoU)
                        log_string('-' * len(s), self.log_out)
                        log_string(s, self.log_out)
                        log_string('-' * len(s) + '\n', self.log_out)

                    self.sess.close()
                    return

                self.sess.run(dataset.test_init_op)
                epoch_id += 1
                step_id = 0
                continue

    def load_evaluation_points(self, file_path):
        data = read_ply(file_path)
        return np.vstack((data['x'], data['y'], data['z'])).T














# import os
# import time
# import glob
# import numpy as np
# import tensorflow as tf
# from os import makedirs
# from os.path import exists, join, basename
# from helper_ply import read_ply
# from sklearn.metrics import precision_recall_fscore_support

# # Replace this with your actual ignore label index
# IGNORE_LABEL = 0  

# class ModelTester(object):
#     def __init__(self, model, dataset, restore_snap=None):
#         # TensorFlow Saver
#         vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
#         self.saver = tf.train.Saver(vars, max_to_keep=100)

#         # TF Session
#         config = tf.ConfigProto()
#         config.gpu_options.allow_growth = True
#         self.sess = tf.Session(config=config)
#         self.sess.run(tf.global_variables_initializer())

#         if restore_snap:
#             if restore_snap.endswith('.meta'):
#                 restore_snap = restore_snap[:-5]
#             self.saver.restore(self.sess, restore_snap)
#             print("Model restored from {}".format(restore_snap))

#         # Softmax logits
#         self.prob_logits = tf.nn.softmax(model.logits)

#         # Vote buffer
#         self.test_probs = [
#             np.zeros((tree.data.shape[0], model.config.num_classes), dtype=np.float32)
#             for tree in dataset.input_trees['test']
#         ]

#         self.log_out = open('log_test_{}.txt'.format(dataset.name), 'a')

#     def log_string(self, s):
#         self.log_out.write(s + '\n')
#         self.log_out.flush()
#         print(s)

#     @staticmethod
#     def load_evaluation_points(ply_path):
#         data = read_ply(ply_path)
#         return np.vstack((data['x'], data['y'], data['z'])).T

#     def test(self, model, dataset, num_votes=100):
#         smooth = 0.98
#         self.sess.run(dataset.test_init_op)

#         # dirs
#         tag      = time.strftime('Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
#         test_dir = join('test', tag)
#         pred_dir = join(test_dir, 'predictions')
#         gt_dir   = dataset.gt_labels_dir
#         for d in (pred_dir,):
#             if not exists(d):
#                 makedirs(d)

#         step, epoch = 0, 0
#         last_min = -0.5

#         while last_min < num_votes:
#             try:
#                 probs_batch, inds_batch, cloud_batch = self.sess.run(
#                     (self.prob_logits,
#                      model.inputs['input_inds'],
#                      model.inputs['cloud_inds']),
#                     {model.is_training: False}
#                 )

#                 B, N, C = model.config.val_batch_size, model.config.num_points, model.config.num_classes
#                 probs_batch = probs_batch.reshape((B, N, C))

#                 for b in range(B):
#                     idxs   = inds_batch[b]
#                     c_idx  = cloud_batch[b][0]
#                     self.test_probs[c_idx][idxs] = (
#                         smooth * self.test_probs[c_idx][idxs]
#                         + (1 - smooth) * probs_batch[b]
#                     )

#                 step += 1
#                 minpos = np.min(dataset.min_possibility['test'])
#                 self.log_string('Epoch {:3d}, step {:3d}, min poss={:.1f}'.format(epoch, step, minpos))

#             except tf.errors.OutOfRangeError:
#                 new_min = np.min(dataset.min_possibility['test'])
#                 self.log_string('Epoch {:3d} end, min poss={:.1f}'.format(epoch, new_min))

#                 if last_min + 4 < new_min:
#                     self.log_string('Reproject & save preds…')
#                     last_min = new_min

#                     # save .labels
#                     for i, ply_path in enumerate(dataset.test_files):
#                         pts      = self.load_evaluation_points(ply_path)
#                         proj_idx = dataset.test_proj[i]
#                         probs    = self.test_probs[i][proj_idx]

#                         # zero-out ignored
#                         P = probs.copy()
#                         for li, lv in enumerate(dataset.label_values):
#                             if lv in dataset.ignored_labels:
#                                 P[:, li] = 0

#                         preds = dataset.label_values[np.argmax(P, axis=1)]
#                         name  = basename(ply_path).replace('.ply', '.labels')
#                         outf  = join(pred_dir, name)
#                         np.savetxt(outf, preds, fmt='%d')
#                         self.log_string('{} saved'.format(outf))

#                     # metric computation
#                     self.log_string('Computing precision/recall/F1…')
#                     all_p, all_t = [], []

#                     # loop only .labels
#                     for gt_path in sorted(glob.glob(join(gt_dir, '*.labels'))):
#                         name = basename(gt_path)
#                         pred_path = join(pred_dir, name)
#                         if not exists(pred_path):
#                             raise RuntimeError('Missing pred for {}'.format(name))

#                         p = np.loadtxt(pred_path, dtype=int)
#                         t = np.loadtxt(gt_path,  dtype=int)
#                         mask = t != IGNORE_LABEL
#                         all_p.append(p[mask])
#                         all_t.append(t[mask])

#                     all_p = np.concatenate(all_p)
#                     all_t = np.concatenate(all_t)

#                     prec, rec, f1, sup = precision_recall_fscore_support(
#                         all_t, all_p,
#                         labels=dataset.label_values,
#                         zero_division=0
#                     )
#                     macro = np.mean(f1)
#                     _, _, micro_f1, _ = precision_recall_fscore_support(
#                         all_t, all_p,
#                         average='micro',
#                         zero_division=0
#                     )

#                     self.log_string('=== Final RandLA-Net Eval ===')
#                     for cls, P, R, F, S in zip(dataset.label_values, prec, rec, f1, sup):
#                         self.log_string('Class {}: P={:.4f}, R={:.4f}, F1={:.4f}, N={}'.format(
#                             cls, P, R, F, S))
#                     self.log_string('Macro-F1: {:.4f}'.format(macro))
#                     self.log_string('Micro-F1: {:.4f}'.format(micro_f1))

#                     self.sess.close()
#                     return

#                 # next epoch
#                 self.sess.run(dataset.test_init_op)
#                 epoch += 1
#                 step   = 0

#         # end voting loop
