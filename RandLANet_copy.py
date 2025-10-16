# from os.path import exists, join
# from os import makedirs
# from sklearn.metrics import confusion_matrix
# from helper_tool import DataProcessing as DP
# import tensorflow as tf
# import numpy as np
# import helper_tf_util
# import time


# def log_out(out_str, f_out):
#     f_out.write(out_str + '\n')
#     f_out.flush()
#     print(out_str)


# class Network:
#     def __init__(self, dataset, config):
#         flat_inputs = dataset.flat_inputs
#         self.config = config
#         # Path of the result folder
#         if self.config.saving:
#             if self.config.saving_path is None:
#                 self.saving_path = time.strftime('results/Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
#             else:
#                 self.saving_path = self.config.saving_path
#             makedirs(self.saving_path) if not exists(self.saving_path) else None

#         with tf.variable_scope('inputs'):
#             self.inputs = dict()
#             num_layers = self.config.num_layers
#             self.inputs['xyz'] = flat_inputs[:num_layers]
#             self.inputs['neigh_idx'] = flat_inputs[num_layers: 2 * num_layers]
#             self.inputs['sub_idx'] = flat_inputs[2 * num_layers:3 * num_layers]
#             self.inputs['interp_idx'] = flat_inputs[3 * num_layers:4 * num_layers]
#             self.inputs['features'] = flat_inputs[4 * num_layers]
#             self.inputs['labels'] = flat_inputs[4 * num_layers + 1]
#             self.inputs['input_inds'] = flat_inputs[4 * num_layers + 2]
#             self.inputs['cloud_inds'] = flat_inputs[4 * num_layers + 3]

#             self.labels = self.inputs['labels']
#             self.is_training = tf.placeholder(tf.bool, shape=())
#             self.training_step = 1
#             self.training_epoch = 0
#             self.correct_prediction = 0
#             self.accuracy = 0
#             self.mIou_list = [0]
#             self.class_weights = DP.get_class_weights(dataset.name)
#             self.Log_file = open('log_train_' + dataset.name + str(dataset.val_split) + '.txt', 'a')

#         with tf.variable_scope('layers'):
#             self.logits = self.inference(self.inputs, self.is_training)

#         #####################################################################
#         # Ignore the invalid point (unlabeled) when calculating the loss #
#         #####################################################################
#         with tf.variable_scope('loss'):
#             self.logits = tf.reshape(self.logits, [-1, config.num_classes])
#             self.labels = tf.reshape(self.labels, [-1])

#             # Boolean mask of points that should be ignored
#             ignored_bool = tf.zeros_like(self.labels, dtype=tf.bool)
#             for ign_label in self.config.ignored_label_inds:
#                 ignored_bool = tf.logical_or(ignored_bool, tf.equal(self.labels, ign_label))

#             # Collect logits and labels that are not ignored
#             valid_idx = tf.squeeze(tf.where(tf.logical_not(ignored_bool)))
#             valid_logits = tf.gather(self.logits, valid_idx, axis=0)
#             valid_labels_init = tf.gather(self.labels, valid_idx, axis=0)

#             # Reduce label values in the range of logit shape
#             reducing_list = tf.range(self.config.num_classes, dtype=tf.int32)
#             inserted_value = tf.zeros((1,), dtype=tf.int32)
#             for ign_label in self.config.ignored_label_inds:
#                 reducing_list = tf.concat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
#             valid_labels = tf.gather(reducing_list, valid_labels_init)

#             self.loss = self.get_loss(valid_logits, valid_labels, self.class_weights)

#         with tf.variable_scope('optimizer'):
#             self.learning_rate = tf.Variable(config.learning_rate, trainable=False, name='learning_rate')
#             self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
#             self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

#         with tf.variable_scope('results'):
#             self.correct_prediction = tf.nn.in_top_k(valid_logits, valid_labels, 1)
#             self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
#             self.prob_logits = tf.nn.softmax(self.logits)

#             tf.summary.scalar('learning_rate', self.learning_rate)
#             tf.summary.scalar('loss', self.loss)
#             tf.summary.scalar('accuracy', self.accuracy)

#         my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
#         self.saver = tf.train.Saver(my_vars, max_to_keep=100)
#         c_proto = tf.ConfigProto()
#         c_proto.gpu_options.allow_growth = True
#         self.sess = tf.Session(config=c_proto)
#         self.merged = tf.summary.merge_all()
#         self.train_writer = tf.summary.FileWriter(config.train_sum_dir, self.sess.graph)
#         self.sess.run(tf.global_variables_initializer())

#     def inference(self, inputs, is_training):

#         d_out = self.config.d_out
#         feature = inputs['features']
#         feature = tf.layers.dense(feature, 8, activation=None, name='fc0')
#         feature = tf.nn.leaky_relu(tf.layers.batch_normalization(feature, -1, 0.99, 1e-6, training=is_training))
#         feature = tf.expand_dims(feature, axis=2)

#         # ###########################Encoder############################
#         f_encoder_list = []
#         for i in range(self.config.num_layers):
#             f_encoder_i = self.dilated_res_block(feature, inputs['xyz'][i], inputs['neigh_idx'][i], d_out[i],
#                                                  'Encoder_layer_' + str(i), is_training)
#             f_sampled_i = self.random_sample(f_encoder_i, inputs['sub_idx'][i])
#             feature = f_sampled_i
#             if i == 0:
#                 f_encoder_list.append(f_encoder_i)
#             f_encoder_list.append(f_sampled_i)
#         # ###########################Encoder############################

#         feature = helper_tf_util.conv2d(f_encoder_list[-1], f_encoder_list[-1].get_shape()[3].value, [1, 1],
#                                         'decoder_0',
#                                         [1, 1], 'VALID', True, is_training)

#         # ###########################Decoder############################
#         f_decoder_list = []
#         for j in range(self.config.num_layers):
#             f_interp_i = self.nearest_interpolation(feature, inputs['interp_idx'][-j - 1])
#             f_decoder_i = helper_tf_util.conv2d_transpose(tf.concat([f_encoder_list[-j - 2], f_interp_i], axis=3),
#                                                           f_encoder_list[-j - 2].get_shape()[-1].value, [1, 1],
#                                                           'Decoder_layer_' + str(j), [1, 1], 'VALID', bn=True,
#                                                           is_training=is_training)
#             feature = f_decoder_i
#             f_decoder_list.append(f_decoder_i)
#         # ###########################Decoder############################

#         f_layer_fc1 = helper_tf_util.conv2d(f_decoder_list[-1], 64, [1, 1], 'fc1', [1, 1], 'VALID', True, is_training)
#         f_layer_fc2 = helper_tf_util.conv2d(f_layer_fc1, 32, [1, 1], 'fc2', [1, 1], 'VALID', True, is_training)
#         f_layer_drop = helper_tf_util.dropout(f_layer_fc2, keep_prob=0.5, is_training=is_training, scope='dp1')
#         f_layer_fc3 = helper_tf_util.conv2d(f_layer_drop, self.config.num_classes, [1, 1], 'fc', [1, 1], 'VALID', False,
#                                             is_training, activation_fn=None)
#         f_out = tf.squeeze(f_layer_fc3, [2])
#         return f_out

#     def train(self, dataset):
#         log_out('****EPOCH {}****'.format(self.training_epoch), self.Log_file)
#         self.sess.run(dataset.train_init_op)
#         while self.training_epoch < self.config.max_epoch:
#             t_start = time.time()
#             try:
#                 ops = [self.train_op,
#                        self.extra_update_ops,
#                        self.merged,
#                        self.loss,
#                        self.logits,
#                        self.labels,
#                        self.accuracy]
#                 _, _, summary, l_out, probs, labels, acc = self.sess.run(ops, {self.is_training: True})
#                 self.train_writer.add_summary(summary, self.training_step)
#                 t_end = time.time()
#                 if self.training_step % 50 == 0:
#                     message = 'Step {:08d} L_out={:5.3f} Acc={:4.2f} ''---{:8.2f} ms/batch'
#                     log_out(message.format(self.training_step, l_out, acc, 1000 * (t_end - t_start)), self.Log_file)
#                 self.training_step += 1

#             except tf.errors.OutOfRangeError:

#                 m_iou = self.evaluate(dataset)
#                 if m_iou > np.max(self.mIou_list):
#                     # Save the best model
#                     snapshot_directory = join(self.saving_path, 'snapshots')
#                     makedirs(snapshot_directory) if not exists(snapshot_directory) else None
#                     self.saver.save(self.sess, snapshot_directory + '/snap', global_step=self.training_step)
#                 self.mIou_list.append(m_iou)
#                 log_out('Best m_IoU is: {:5.3f}'.format(max(self.mIou_list)), self.Log_file)

#                 self.training_epoch += 1
#                 self.sess.run(dataset.train_init_op)
#                 # Update learning rate
#                 op = self.learning_rate.assign(tf.multiply(self.learning_rate,
#                                                            self.config.lr_decays[self.training_epoch]))
#                 self.sess.run(op)
#                 log_out('****EPOCH {}****'.format(self.training_epoch), self.Log_file)

#             except tf.errors.InvalidArgumentError as e:

#                 print('Caught a NaN error :')
#                 print(e.error_code)
#                 print(e.message)
#                 print(e.op)
#                 print(e.op.name)
#                 print([t.name for t in e.op.inputs])
#                 print([t.name for t in e.op.outputs])

#                 a = 1 / 0

#         print('finished')
#         self.sess.close()

#     def evaluate(self, dataset):

#         # Initialise iterator with validation data
#         self.sess.run(dataset.val_init_op)

#         gt_classes = [0 for _ in range(self.config.num_classes)]
#         positive_classes = [0 for _ in range(self.config.num_classes)]
#         true_positive_classes = [0 for _ in range(self.config.num_classes)]
#         val_total_correct = 0
#         val_total_seen = 0

#         for step_id in range(self.config.val_steps):
#             if step_id % 50 == 0:
#                 print(str(step_id) + ' / ' + str(self.config.val_steps))
#             try:
#                 ops = (self.prob_logits, self.labels, self.accuracy)
#                 stacked_prob, labels, acc = self.sess.run(ops, {self.is_training: False})
#                 pred = np.argmax(stacked_prob, 1)
#                 if not self.config.ignored_label_inds:
#                     pred_valid = pred
#                     labels_valid = labels
#                 else:
#                     invalid_idx = np.where(labels == self.config.ignored_label_inds)[0]
#                     labels_valid = np.delete(labels, invalid_idx)
#                     labels_valid = labels_valid - 1
#                     pred_valid = np.delete(pred, invalid_idx)

#                 correct = np.sum(pred_valid == labels_valid)
#                 val_total_correct += correct
#                 val_total_seen += len(labels_valid)

#                 conf_matrix = confusion_matrix(labels_valid, pred_valid, np.arange(0, self.config.num_classes, 1))
#                 gt_classes += np.sum(conf_matrix, axis=1)
#                 positive_classes += np.sum(conf_matrix, axis=0)
#                 true_positive_classes += np.diagonal(conf_matrix)

#             except tf.errors.OutOfRangeError:
#                 break

#         iou_list = []
#         for n in range(0, self.config.num_classes, 1):
#             iou = true_positive_classes[n] / float(gt_classes[n] + positive_classes[n] - true_positive_classes[n])
#             iou_list.append(iou)
#         mean_iou = sum(iou_list) / float(self.config.num_classes)

#         log_out('eval accuracy: {}'.format(val_total_correct / float(val_total_seen)), self.Log_file)
#         log_out('mean IOU:{}'.format(mean_iou), self.Log_file)

#         mean_iou = 100 * mean_iou
#         log_out('Mean IoU = {:.1f}%'.format(mean_iou), self.Log_file)
#         s = '{:5.2f} | '.format(mean_iou)
#         for IoU in iou_list:
#             s += '{:5.2f} '.format(100 * IoU)
#         log_out('-' * len(s), self.Log_file)
#         log_out(s, self.Log_file)
#         log_out('-' * len(s) + '\n', self.Log_file)
#         return mean_iou

#     def get_loss(self, logits, labels, pre_cal_weights):
#         # calculate the weighted cross entropy according to the inverse frequency
#         class_weights = tf.convert_to_tensor(pre_cal_weights, dtype=tf.float32)
#         one_hot_labels = tf.one_hot(labels, depth=self.config.num_classes)
#         weights = tf.reduce_sum(class_weights * one_hot_labels, axis=1)
#         unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_labels)
#         weighted_losses = unweighted_losses * weights
#         output_loss = tf.reduce_mean(weighted_losses)
#         return output_loss

#     def dilated_res_block(self, feature, xyz, neigh_idx, d_out, name, is_training):
#         f_pc = helper_tf_util.conv2d(feature, d_out // 2, [1, 1], name + 'mlp1', [1, 1], 'VALID', True, is_training)
#         f_pc = self.building_block(xyz, f_pc, neigh_idx, d_out, name + 'LFA', is_training)
#         f_pc = helper_tf_util.conv2d(f_pc, d_out * 2, [1, 1], name + 'mlp2', [1, 1], 'VALID', True, is_training,
#                                      activation_fn=None)
#         shortcut = helper_tf_util.conv2d(feature, d_out * 2, [1, 1], name + 'shortcut', [1, 1], 'VALID',
#                                          activation_fn=None, bn=True, is_training=is_training)
#         return tf.nn.leaky_relu(f_pc + shortcut)

#     def building_block(self, xyz, feature, neigh_idx, d_out, name, is_training):
#         d_in = feature.get_shape()[-1].value
#         f_xyz = self.relative_pos_encoding(xyz, neigh_idx)
#         f_xyz = helper_tf_util.conv2d(f_xyz, d_in, [1, 1], name + 'mlp1', [1, 1], 'VALID', True, is_training)
#         f_neighbours = self.gather_neighbour(tf.squeeze(feature, axis=2), neigh_idx)
#         f_concat = tf.concat([f_neighbours, f_xyz], axis=-1)
#         f_pc_agg = self.att_pooling(f_concat, d_out // 2, name + 'att_pooling_1', is_training)

#         f_xyz = helper_tf_util.conv2d(f_xyz, d_out // 2, [1, 1], name + 'mlp2', [1, 1], 'VALID', True, is_training)
#         f_neighbours = self.gather_neighbour(tf.squeeze(f_pc_agg, axis=2), neigh_idx)
#         f_concat = tf.concat([f_neighbours, f_xyz], axis=-1)
#         f_pc_agg = self.att_pooling(f_concat, d_out, name + 'att_pooling_2', is_training)
#         return f_pc_agg

#     def relative_pos_encoding(self, xyz, neigh_idx):
#         neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)
#         xyz_tile = tf.tile(tf.expand_dims(xyz, axis=2), [1, 1, tf.shape(neigh_idx)[-1], 1])
#         relative_xyz = xyz_tile - neighbor_xyz
#         relative_dis = tf.sqrt(tf.reduce_sum(tf.square(relative_xyz), axis=-1, keepdims=True))
#         relative_feature = tf.concat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], axis=-1)
#         return relative_feature

#     @staticmethod
#     def random_sample(feature, pool_idx):
#         feature = tf.squeeze(feature, axis=2)
#         num_neigh = tf.shape(pool_idx)[-1]
#         d = feature.get_shape()[-1]
#         batch_size = tf.shape(pool_idx)[0]
#         pool_idx = tf.reshape(pool_idx, [batch_size, -1])
#         pool_features = tf.batch_gather(feature, pool_idx)
#         pool_features = tf.reshape(pool_features, [batch_size, -1, num_neigh, d])
#         pool_features = tf.reduce_max(pool_features, axis=2, keepdims=True)
#         return pool_features

#     @staticmethod
#     def nearest_interpolation(feature, interp_idx):
#         feature = tf.squeeze(feature, axis=2)
#         batch_size = tf.shape(interp_idx)[0]
#         up_num_points = tf.shape(interp_idx)[1]
#         interp_idx = tf.reshape(interp_idx, [batch_size, up_num_points])
#         interpolated_features = tf.batch_gather(feature, interp_idx)
#         interpolated_features = tf.expand_dims(interpolated_features, axis=2)
#         return interpolated_features

#     @staticmethod
#     def gather_neighbour(pc, neighbor_idx):
#         # gather the coordinates or features of neighboring points
#         batch_size = tf.shape(pc)[0]
#         num_points = tf.shape(pc)[1]
#         d = pc.get_shape()[2].value
#         index_input = tf.reshape(neighbor_idx, shape=[batch_size, -1])
#         features = tf.batch_gather(pc, index_input)
#         features = tf.reshape(features, [batch_size, num_points, tf.shape(neighbor_idx)[-1], d])
#         return features

#     @staticmethod
#     def att_pooling(feature_set, d_out, name, is_training):
#         batch_size = tf.shape(feature_set)[0]
#         num_points = tf.shape(feature_set)[1]
#         num_neigh = tf.shape(feature_set)[2]
#         d = feature_set.get_shape()[3].value
#         f_reshaped = tf.reshape(feature_set, shape=[-1, num_neigh, d])
#         att_activation = tf.layers.dense(f_reshaped, d, activation=None, use_bias=False, name=name + 'fc')
#         att_scores = tf.nn.softmax(att_activation, axis=1)
#         f_agg = f_reshaped * att_scores
#         f_agg = tf.reduce_sum(f_agg, axis=1)
#         f_agg = tf.reshape(f_agg, [batch_size, num_points, 1, d])
#         f_agg = helper_tf_util.conv2d(f_agg, d_out, [1, 1], name + 'mlp', [1, 1], 'VALID', True, is_training)
#         return f_agg


# from os.path import exists, join
# from os import makedirs
# from sklearn.metrics import confusion_matrix
# from helper_tool import DataProcessing as DP
# import tensorflow as tf
# import numpy as np
# import helper_tf_util
# import time
# import math

# tf.reset_default_graph()

# def log_out(out_str, f_out):
#     f_out.write(out_str + '\n')
#     f_out.flush()
#     print(out_str)

# class Network:
#     def __init__(self, dataset, config):
#         self.config = config
#         num_features = config.num_features
        
#         # Setup logging
#         self.Log_file = open('log_train_' + dataset.name + str(dataset.val_split) + '.txt', 'a')
#         log_out("===== NETWORK INITIALIZATION =====", self.Log_file)
        
#         flat_inputs = dataset.flat_inputs
#         L = config.num_layers
        
#         # Dataset verification
#         log_out("Verifying dataset structure:", self.Log_file)
#         log_out("Total inputs: %d" % len(flat_inputs), self.Log_file)
#         log_out("Expected: %d items (xyz: %d, neigh_idx: %d, etc.)" % 
#                (5*L + 4, L, L), self.Log_file)
        
#         for i in range(L):
#             xyz_shape = flat_inputs[i].shape
#             neigh_shape = flat_inputs[L+i].shape
#             log_out("Layer %d: XYZ shape=%s, Neigh_idx shape=%s" % 
#                    (i, str(xyz_shape), str(neigh_shape)), self.Log_file)
            
#             if xyz_shape[1] != neigh_shape[1]:
#                 log_out("ERROR: Point count mismatch in layer %d: %d vs %d" % 
#                        (i, xyz_shape[1], neigh_shape[1]), self.Log_file)
#                 raise ValueError("Layer %d: Point count mismatch" % i)
        
#         # Saving setup
#         if self.config.saving:
#             self.saving_path = config.saving_path or time.strftime('results/Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
#             if not exists(self.saving_path):
#                 makedirs(self.saving_path)
#             log_out("Saving to: " + self.saving_path, self.Log_file)

#         with tf.variable_scope('inputs'):
#             self.inputs = {
#                 'xyz': flat_inputs[:L],
#                 'neigh_idx': flat_inputs[L:2*L],
#                 'sub_idx': flat_inputs[2*L:3*L],
#                 'interp_idx': flat_inputs[3*L:4*L],
#                 'features': flat_inputs[4*L],
#                 'labels': flat_inputs[4*L+1],
#                 'input_inds': flat_inputs[4*L+2],
#                 'cloud_inds': flat_inputs[4*L+3],
#             }
#             self.inputs['features'].set_shape([None, None, num_features])
            
#             self.labels = self.inputs['labels']
#             self.is_training = tf.placeholder(tf.bool, shape=())
#             self.training_step = 1
#             self.training_epoch = 0
#             self.correct_prediction = 0
#             self.accuracy = 0
#             self.mIou_list = [0]
#             self.class_weights = DP.get_class_weights(dataset.name)
            
#             # Store point coordinates at each layer
#             self.encoder_xyz = [self.inputs['xyz'][0]]
#             self.sample_indices = []

#         with tf.variable_scope('layers'):
#             self.logits = self.inference(self.inputs, self.is_training)

#         with tf.variable_scope('loss'):
#             self.logits = tf.reshape(self.logits, [-1, config.num_classes])
#             self.labels = tf.reshape(self.labels, [-1])
            
#             # Handle ignored labels
#             ignored_bool = tf.zeros_like(self.labels, dtype=tf.bool)
#             for ign_label in self.config.ignored_label_inds:
#                 ignored_bool = tf.logical_or(ignored_bool, tf.equal(self.labels, ign_label))
            
#             valid_idx = tf.squeeze(tf.where(tf.logical_not(ignored_bool)))
#             valid_logits = tf.gather(self.logits, valid_idx, axis=0)
#             valid_labels_init = tf.gather(self.labels, valid_idx, axis=0)
            
#             # Adjust labels for ignored classes
#             reducing_list = tf.range(self.config.num_classes, dtype=tf.int32)
#             inserted_value = tf.zeros((1,), dtype=tf.int32)
#             for ign_label in self.config.ignored_label_inds:
#                 reducing_list = tf.concat([
#                     reducing_list[:ign_label], 
#                     inserted_value, 
#                     reducing_list[ign_label:]
#                 ], 0)
#             valid_labels = tf.gather(reducing_list, valid_labels_init)
            
#             self.loss = self.get_loss(valid_logits, valid_labels, self.class_weights)

#         with tf.variable_scope('optimizer'):
#             self.learning_rate = tf.Variable(config.learning_rate, trainable=False, name='learning_rate')
#             self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
#             self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

#         with tf.variable_scope('results'):
#             self.correct_prediction = tf.nn.in_top_k(valid_logits, valid_labels, 1)
#             self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
#             self.prob_logits = tf.nn.softmax(self.logits)
            
#             tf.summary.scalar('learning_rate', self.learning_rate)
#             tf.summary.scalar('loss', self.loss)
#             tf.summary.scalar('accuracy', self.accuracy)

#         # Initialize session
#         my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
#         self.saver = tf.train.Saver(my_vars, max_to_keep=100)
#         c_proto = tf.ConfigProto()
#         c_proto.gpu_options.allow_growth = True
#         self.sess = tf.Session(config=c_proto)
#         self.merged = tf.summary.merge_all()
#         self.train_writer = tf.summary.FileWriter(config.train_sum_dir, self.sess.graph)
#         self.sess.run(tf.global_variables_initializer())
        
#         log_out("Network initialized successfully", self.Log_file)

#     def inference(self, inputs, is_training):
#         d_out = self.config.d_out
        
#         # ================== Initial Feature Processing ==================
#         feature = inputs['features']
#         feature = tf.layers.dense(feature, 8, name='fc0')
#         feature = tf.nn.leaky_relu(
#             tf.layers.batch_normalization(
#                 feature, axis=-1, momentum=0.99, epsilon=1e-6, training=is_training
#             )
#         )
#         feature = tf.expand_dims(feature, axis=2)
        
#         # ================== ENCODER ==================
#         f_encoder_list = [feature]
#         for i in range(self.config.num_layers):
#             # Transformer block
#             f_encoder_i = self.transformer_res_block(
#                 self.encoder_xyz[i],
#                 feature,
#                 inputs['neigh_idx'][i],
#                 d_out[i],
#                 "Encoder_%d" % i,
#                 is_training
#             )
            
#             # Downsampling
#             f_sampled_i, xyz_sampled, sample_idx = self.learnable_sample(
#                 f_encoder_i,
#                 self.encoder_xyz[i],
#                 self.config.sub_sampling_ratio[i],
#                 "Sample_{}".format(i),
#                 is_training
#             )
            
#             # Store for decoder
#             self.encoder_xyz.append(xyz_sampled)
#             self.sample_indices.append(sample_idx)
#             feature = f_sampled_i
#             f_encoder_list.append(f_sampled_i)
        
#         # ================== BOTTLENECK ==================
#         feature = helper_tf_util.conv2d(
#             f_encoder_list[-1],
#             f_encoder_list[-1].get_shape()[-1].value,
#             [1, 1],
#             'decoder_0',
#             [1, 1],
#             'VALID',
#             True,
#             is_training
#         )
        
#         # ================== DECODER ==================
#         f_decoder_list = []
#         for j in range(self.config.num_layers):
#             # Get skip feature
#             skip_feat = f_encoder_list[-j-2]
            
#             # Upsampling
#             f_interp = self.feature_upsampling(
#                 feature,
#                 skip_feat,
#                 self.sample_indices[-j-1],
#                 "Upsample_%d" % j,
#                 is_training
#             )
            
#             # Get corresponding XYZ and neighbor indices
#             xyz_idx = self.config.num_layers - j - 1
#             neigh_idx_idx = self.config.num_layers - j - 1
            
#             # Transformer block
#             f_dec = self.transformer_res_block(
#                 self.encoder_xyz[xyz_idx],
#                 tf.concat([f_encoder_list[xyz_idx], f_interp], axis=3),
#                 inputs['neigh_idx'][neigh_idx_idx],
#                 f_encoder_list[xyz_idx].get_shape()[-1].value,
#                 "Decoder_%d" % j,
#                 is_training
#             )
#             feature = f_dec
#             f_decoder_list.append(f_dec)
        
#         # ================== FINAL CLASSIFICATION ==================
#         f_layer_fc1 = helper_tf_util.conv2d(
#             f_decoder_list[-1], 64, [1, 1], 'fc1', [1, 1], 'VALID', True, is_training
#         )
#         f_layer_fc2 = helper_tf_util.conv2d(
#             f_layer_fc1, 32, [1, 1], 'fc2', [1, 1], 'VALID', True, is_training
#         )
#         f_layer_drop = helper_tf_util.dropout(
#             f_layer_fc2, keep_prob=0.5, is_training=is_training, scope='dp1'
#         )
#         f_layer_fc3 = helper_tf_util.conv2d(
#             f_layer_drop,
#             self.config.num_classes,
#             [1, 1],
#             'fc', [1, 1],
#             'VALID',
#             False,
#             is_training,
#             activation_fn=None
#         )
#         f_out = tf.squeeze(f_layer_fc3, [2])
#         return f_out

#     def transformer_res_block(self, xyz, feature, neigh_idx, d_out, name, is_training):
#         # 1. Residual shortcut
#         shortcut = feature
#         C_in = shortcut.get_shape().as_list()[-1]
        
#         # Project shortcut if needed
#         if C_in != d_out * 2:
#             shortcut = helper_tf_util.conv2d(
#                 shortcut, d_out * 2, [1, 1],
#                 name + "_shortcut", [1, 1], 'VALID',
#                 activation_fn=None, bn=True, is_training=is_training
#             )
        
#         # 2. Transformer aggregation
#         f_trans = self.transformer_aggregation(
#             xyz, feature, neigh_idx, d_out, name, is_training
#         )
        
#         # 3. MLP output
#         f_trans = helper_tf_util.conv2d(
#             f_trans, d_out * 2, [1, 1],
#             name + "mlp_out", [1, 1], 'VALID',
#             True, is_training, activation_fn=None
#         )
        
#         # 4. Add & activate
#         added = f_trans + shortcut
#         output = tf.nn.leaky_relu(added)
#         return output

#     def transformer_aggregation(self, xyz, feature, neigh_idx, d_out, name, is_training):
#         with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
#             # 1. Prepare center features
#             center_feat = tf.squeeze(feature, axis=2)
#             C_in = center_feat.get_shape().as_list()[-1]
#             center_feat.set_shape([None, None, C_in])
            
#             # 2. Gather neighbors
#             neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)
#             neighbor_feat = self.gather_neighbour(center_feat, neigh_idx)
            
#             # 3. Positional encoding
#             center_xyz = tf.expand_dims(xyz, axis=2)
#             rel_pos = neighbor_xyz - center_xyz
#             rel_dist = tf.sqrt(tf.reduce_sum(tf.square(rel_pos), axis=-1, keepdims=True))
#             pos_enc = tf.concat([rel_dist, rel_pos], axis=-1)
#             pos_enc = tf.layers.dense(
#                 pos_enc,
#                 d_out // 4,
#                 activation=tf.nn.leaky_relu,
#                 name='pos_enc'
#             )
            
#             # 4. Multi-head attention setup
#             num_heads = 4
#             head_dim = d_out // num_heads
            
#             # Query projection
#             Q = tf.layers.dense(center_feat, d_out, name='query')
#             Q = tf.reshape(Q, [tf.shape(Q)[0], -1, num_heads, head_dim])
#             Q = tf.transpose(Q, [0, 2, 1, 3])
            
#             # Key projection
#             K_feat = tf.layers.dense(neighbor_feat, d_out, name='key')
#             K_pos = tf.layers.dense(pos_enc, d_out, name='pos_key')
#             K_total = K_feat + K_pos
#             K_total = tf.reshape(K_total, [tf.shape(K_total)[0], tf.shape(K_total)[1], 
#                                          tf.shape(K_total)[2], num_heads, head_dim])
#             K_total = tf.transpose(K_total, [0, 3, 1, 2, 4])
            
#             # Value projection
#             V_feat = tf.layers.dense(neighbor_feat, d_out, name='value')
#             V_pos = tf.layers.dense(pos_enc, d_out, name='pos_value')
#             V_total = tf.reshape(V_feat + V_pos, [tf.shape(V_feat)[0], tf.shape(V_feat)[1], 
#                                                 tf.shape(V_feat)[2], num_heads, head_dim])
#             V_total = tf.transpose(V_total, [0, 3, 1, 2, 4])
            
#             # 5. Attention mechanism
#             Q_expanded = tf.expand_dims(Q, axis=3)
#             attn_scores = tf.matmul(Q_expanded, K_total, transpose_b=True) / tf.sqrt(tf.cast(head_dim, tf.float32))
#             attn_w = tf.nn.softmax(attn_scores, axis=-1)
#             attn_out = tf.matmul(attn_w, V_total)
#             attn_out = tf.squeeze(attn_out, axis=3)
#             attn_out = tf.transpose(attn_out, [0, 2, 1, 3])
#             attn_out = tf.reshape(attn_out, [tf.shape(attn_out)[0], -1, d_out])
#             attn_out = tf.layers.dense(attn_out, d_out, name='out_proj')
            
#             # 6. Residual connection
#             if C_in != d_out:
#                 center_feat_proj = tf.layers.dense(center_feat, d_out, name='res_proj')
#             else:
#                 center_feat_proj = center_feat
                
#             combined = center_feat_proj + attn_out
            
#             # 7. Feed-forward network
#             ff1 = tf.layers.dense(combined, d_out * 2, activation=tf.nn.leaky_relu, name='ffn1')
#             ff2 = tf.layers.dense(ff1, d_out, name='ffn2')
#             ff_out = combined + ff2
            
#             return tf.expand_dims(ff_out, axis=2)

#     def learnable_sample(self, feature, xyz, sample_ratio, name, is_training):
#         # Flatten features
#         flat_features = tf.squeeze(feature, axis=2)
        
#         # Ensure XYZ has proper rank
#         if len(xyz.shape) == 4 and xyz.shape[2] == 1:
#             xyz = tf.squeeze(xyz, axis=2)
        
#         # Generate importance scores
#         scores = tf.layers.dense(
#             tf.concat([flat_features, xyz], axis=-1),
#             64,
#             activation=tf.nn.leaky_relu,
#             name=name + "_scoring_1"
#         )
#         scores = tf.layers.dense(
#             scores,
#             32,
#             activation=tf.nn.leaky_relu,
#             name=name + "_scoring_2"
#         )
#         scores = tf.layers.dense(
#             scores,
#             1,
#             activation=None,
#             name=name + "_scoring_out"
#         )
#         scores = tf.squeeze(scores, axis=-1)
        
#         # Calculate sample count
#         num_points = tf.shape(xyz)[1]
#         reduction_factor = tf.cast(sample_ratio, tf.float32)
#         k_float = tf.cast(num_points, tf.float32) / reduction_factor
#         k = tf.minimum(tf.cast(k_float, tf.int32), num_points)
#         k = tf.maximum(k, 1)
        
#         _, indices = tf.nn.top_k(scores, k=k, sorted=False)
        
#         # Create batch indices
#         batch_indices = tf.tile(
#             tf.expand_dims(tf.range(tf.shape(xyz)[0]), 1),
#             [1, k]
#         )
#         full_indices = tf.stack([batch_indices, indices], axis=-1)
        
#         # Gather sampled points
#         sampled_xyz = tf.gather_nd(xyz, full_indices)
#         sampled_features = tf.gather_nd(flat_features, full_indices)
#         sampled_features = tf.expand_dims(sampled_features, axis=2)
        
#         return sampled_features, sampled_xyz, indices
    
#     def feature_upsampling(self, current_feat, skip_feat, sample_idx, name, is_training):
#         # Flatten current features
#         current_flat = tf.squeeze(current_feat, axis=2)
#         D1 = current_flat.get_shape().as_list()[-1]
        
#         # Prepare skip features
#         if len(skip_feat.shape) == 4 and skip_feat.shape[2] == 1:
#             skip_flat = tf.squeeze(skip_feat, axis=2)
#         else:
#             skip_flat = skip_feat
#         D2 = skip_flat.get_shape().as_list()[-1]
        
#         # Project to skip feature dimension
#         up_feat = tf.layers.dense(current_flat, D2, name=name + "_proj")
        
#         # Scatter to full resolution
#         batch_size = tf.shape(skip_flat)[0]
#         num_points = tf.shape(skip_flat)[1]
#         feat_dim = D2
        
#         batch_idx = tf.tile(
#             tf.expand_dims(tf.range(batch_size), 1),
#             [1, tf.shape(sample_idx)[1]]
#         )
#         scatter_idx = tf.stack([batch_idx, sample_idx], axis=-1)
        
#         up_feat_scattered = tf.scatter_nd(
#             indices=scatter_idx,
#             updates=up_feat,
#             shape=[batch_size, num_points, feat_dim]
#         )
        
#         # Combine and merge
#         combined = tf.concat([skip_flat, up_feat_scattered], axis=-1)
        
#         merged = tf.layers.dense(
#             combined,
#             D2,
#             activation=tf.nn.leaky_relu,
#             name=name + "_merge"
#         )
        
#         # Restore feature dimension
#         return tf.expand_dims(merged, axis=2)

#     @staticmethod
#     def gather_neighbour(pc, neighbor_idx):
#         batch_size = tf.shape(pc)[0]
#         num_points = tf.shape(pc)[1]
#         feat_dim = pc.get_shape().as_list()[-1]
        
#         # Flatten point cloud
#         pc_flat = tf.reshape(pc, [batch_size * num_points, feat_dim])
        
#         # Create batch offsets
#         batch_offset = tf.reshape(tf.range(batch_size) * num_points, [batch_size, 1, 1])
#         idx_flat = tf.reshape(neighbor_idx + batch_offset, [-1])
        
#         # Gather neighbors
#         neigh_flat = tf.gather(pc_flat, idx_flat)
        
#         # Reshape to original structure
#         K = tf.shape(neighbor_idx)[2]
#         neighbor = tf.reshape(neigh_flat, [batch_size, num_points, K, feat_dim])
        
#         return neighbor

#     def train(self, dataset):
#         log_out("****EPOCH %d****" % self.training_epoch, self.Log_file)
#         self.sess.run(dataset.train_init_op)
#         step_times = []
        
#         while self.training_epoch < self.config.max_epoch:
#             t_start = time.time()
#             try:
#                 ops = [
#                     self.train_op,
#                     self.extra_update_ops,
#                     self.merged,
#                     self.loss,
#                     self.accuracy
#                 ]
#                 _, _, summary, l_out, accuracy_batch = self.sess.run(
#                     ops, {self.is_training: True}
#                 )
#                 self.train_writer.add_summary(summary, self.training_step)
#                 t_end = time.time()
                
#                 # Track step time for performance monitoring
#                 step_time = (t_end - t_start) * 1000  # in ms
#                 step_times.append(step_time)
                
#                 if self.training_step % 10 == 0:
#                     avg_time = np.mean(step_times[-10:]) if len(step_times) > 10 else step_time
#                     message = "Step %08d L_out=%5.3f Acc=%4.2f ---%8.2f ms/batch"
#                     log_out(message % (self.training_step, l_out, accuracy_batch, avg_time), self.Log_file)
                
#                 self.training_step += 1

#             except tf.errors.OutOfRangeError:
#                 # End of epoch
#                 m_iou = self.evaluate(dataset)
#                 if m_iou > np.max(self.mIou_list):
#                     snapshot_directory = join(self.saving_path, 'snapshots')
#                     if not exists(snapshot_directory):
#                         makedirs(snapshot_directory)
#                     self.saver.save(self.sess, join(snapshot_directory, 'snap'), global_step=self.training_step)
#                 self.mIou_list.append(m_iou)
#                 log_out("Best m_IoU is: %5.3f" % max(self.mIou_list), self.Log_file)

#                 self.training_epoch += 1
#                 self.sess.run(dataset.train_init_op)
                
#                 # Update learning rate
#                 op = self.learning_rate.assign(
#                     tf.multiply(self.learning_rate, self.config.lr_decays[self.training_epoch])
#                 )
#                 self.sess.run(op)
#                 log_out("****EPOCH %d****" % self.training_epoch, self.Log_file)

#             except tf.errors.InvalidArgumentError as e:
#                 log_out("Numerical error detected at step %d" % self.training_step, self.Log_file)
#                 log_out("Error details: " + str(e), self.Log_file)
                
#                 # Save crash diagnostics
#                 crash_path = join(self.saving_path, 'crash_snapshot')
#                 self.saver.save(self.sess, crash_path)
#                 log_out("Saved crash snapshot to: " + crash_path, self.Log_file)
#                 log_out("Terminating training", self.Log_file)
#                 break

#         log_out("Training finished", self.Log_file)
#         self.sess.close()

#     def evaluate(self, dataset):
#         self.sess.run(dataset.val_init_op)

#         gt_classes = [0] * self.config.num_classes
#         positive_classes = [0] * self.config.num_classes
#         true_positive_classes = [0] * self.config.num_classes
#         val_total_correct = 0
#         val_total_seen = 0

#         for step_id in range(self.config.val_steps):
#             if step_id % 50 == 0:
#                 log_out("%d / %d" % (step_id, self.config.val_steps), self.Log_file)
#             try:
#                 ops = (self.prob_logits, self.labels)
#                 stacked_prob, labels = self.sess.run(ops, {self.is_training: False})
#                 pred = np.argmax(stacked_prob, 1)
                
#                 if not self.config.ignored_label_inds:
#                     pred_valid = pred
#                     labels_valid = labels
#                 else:
#                     invalid_idx = np.where(labels == self.config.ignored_label_inds)[0]
#                     labels_valid = np.delete(labels, invalid_idx)
#                     labels_valid = labels_valid - 1
#                     pred_valid = np.delete(pred, invalid_idx)

#                 correct = np.sum(pred_valid == labels_valid)
#                 val_total_correct += correct
#                 val_total_seen += len(labels_valid)

#                 conf_matrix = confusion_matrix(labels_valid, pred_valid,
#                                                np.arange(0, self.config.num_classes, 1))
#                 gt_classes += np.sum(conf_matrix, axis=1)
#                 positive_classes += np.sum(conf_matrix, axis=0)
#                 true_positive_classes += np.diagonal(conf_matrix)

#             except tf.errors.OutOfRangeError:
#                 break

#         iou_list = []
#         for n in range(self.config.num_classes):
#             denominator = gt_classes[n] + positive_classes[n] - true_positive_classes[n]
#             iou = true_positive_classes[n] / float(denominator) if denominator > 0 else 0
#             iou_list.append(iou)
            
#         mean_iou = np.mean(iou_list)
#         mean_iou_percent = 100 * mean_iou

#         log_out("Evaluation accuracy: %.4f" % (val_total_correct / float(val_total_seen)), self.Log_file)
#         log_out("Mean IoU: %.4f" % mean_iou, self.Log_file)
#         log_out("Mean IoU = %.1f%%" % mean_iou_percent, self.Log_file)
        
#         s = "%.2f | " % mean_iou_percent
#         for IoU in iou_list:
#             s += "%.2f " % (100 * IoU)
            
#         log_out("-" * len(s), self.Log_file)
#         log_out(s, self.Log_file)
#         log_out("-" * len(s) + "\n", self.Log_file)
        
#         return mean_iou_percent

#     def get_loss(self, logits, labels, pre_cal_weights):
#         class_weights = tf.convert_to_tensor(pre_cal_weights, dtype=tf.float32)
#         one_hot_labels = tf.one_hot(labels, depth=self.config.num_classes)
#         weights = tf.reduce_sum(class_weights * one_hot_labels, axis=1)
#         unweighted_losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=one_hot_labels)
#         weighted_losses = unweighted_losses * weights
#         return tf.reduce_mean(weighted_losses)

#     @staticmethod
#     def nearest_interpolation(feature, interp_idx):
#         feature = tf.squeeze(feature, axis=2)
#         batch_size = tf.shape(interp_idx)[0]
#         up_num_points = tf.shape(interp_idx)[1]
#         interp_idx = tf.reshape(interp_idx, [batch_size, up_num_points])
#         interpolated_features = tf.batch_gather(feature, interp_idx)
#         return tf.expand_dims(interpolated_features, axis=2)



# RandLANetPlus.py
# Novel RandLA-Net++ Architecture with
#   1) Geometry-Adaptive Sampling (GAS)
#   2) Multi-Scale Local Grouping (MSLG)
#   3) Lightweight Local Transformer Aggregation (LTA)
#
# Author: <Your Name>
# Date: 2025-06-06
#
# Dependencies:
#   tensorflow==1.15, numpy, sklearn
# Place this file in the same folder as main_semantic3d.py (or in a subfolder, adjusting imports accordingly).
# RandLANetPlus.py
# Novel RandLA-Net++ Architecture with
#   1) Geometry-Adaptive Sampling (GAS)
#   2) Multi-Scale Local Grouping (MSLG)
#   3) Lightweight Local Transformer Aggregation (LTA)
#
# Author: <Your Name>
# Date: 2025-06-06
#
# Dependencies:
#   tensorflow==1.15, numpy, sklearn
# Place this file in the same folder as main_semantic3d.py (or in a subfolder, adjusting imports accordingly).



import tensorflow as tf
import numpy as np
from os.path import exists, join
from os import makedirs
from sklearn.metrics import confusion_matrix
from helper_tool import DataProcessing as DP  
import time

def debug_print(tensor, message):
    """Debug printing function that preserves gradients"""
    return tf.Print(tensor, [tensor], message=message + ": ", summarize=100)
def knn_blocked(xyz, k, block_size=4096, chunk_size=256):
    B = tf.shape(xyz)[0]
    N = tf.shape(xyz)[1]
    xyz_sq = tf.reduce_sum(tf.square(xyz), axis=-1)

    num_blocks = tf.cast(tf.ceil(tf.cast(N, tf.float32) / block_size), tf.int32)
    blocks = tf.range(num_blocks)

    def process_block(b):
        start = b * block_size
        end = tf.minimum(start + block_size, N)
        xi = xyz[:, start:end, :]
        block_size_i = tf.shape(xi)[1]
        xi_sq = tf.reduce_sum(tf.square(xi), axis=-1)

        num_chunks = tf.cast(tf.ceil(tf.cast(N, tf.float32) / chunk_size), tf.int32)
        top_d = tf.fill([B, block_size_i, k], 1e10)
        top_i = tf.zeros([B, block_size_i, k], tf.int32)

        _, top_d_final, top_i_final = tf.while_loop(
            lambda i, *_: i < num_chunks,
            lambda i, td, ti: [
                i + 1,
                *update_top_k(i, xi, xi_sq, xyz, xyz_sq, td, ti, chunk_size, k)
            ],
            [0, top_d, top_i],
            shape_invariants=[
                tf.TensorShape([]),
                tf.TensorShape([None, None, k]),
                tf.TensorShape([None, None, k])
            ]
        )

        # Pad to full block_size so map_fn can stack correctly
        pad = block_size - block_size_i
        paddings = tf.zeros([B, pad, k], tf.int32)
        return tf.concat([top_i_final, paddings], axis=1)

    def update_top_k(i, xi, xi_sq, xyz, xyz_sq, top_d, top_i, chunk_size, k):
        B = tf.shape(xi)[0]
        block_size_i = tf.shape(xi)[1]
        start_chunk = i * chunk_size
        end_chunk = tf.minimum(start_chunk + chunk_size, N)
        chunk = xyz[:, start_chunk:end_chunk, :]
        chunk_sq = xyz_sq[:, start_chunk:end_chunk]

        ip = tf.matmul(xi, chunk, transpose_b=True)
        dist = xi_sq[:, :, None] + chunk_sq[:, None, :] - 2.0 * ip

        global_idx = tf.range(start_chunk, end_chunk)
        global_idx = tf.tile(global_idx[None, None, :], [B, block_size_i, 1])
        cand_d = tf.concat([top_d, dist], axis=2)
        cand_i = tf.concat([top_i, global_idx], axis=2)

        topk_vals, topk_pos = tf.nn.top_k(-cand_d, k=k)
        topk_vals = -topk_vals
        Bn = B * block_size_i
        flat_idx = tf.reshape(cand_i, [Bn, -1])
        pos_flat = tf.reshape(topk_pos, [Bn, k])

        row = tf.range(Bn)
        row = tf.reshape(tf.tile(row[:, None], [1, k]), [-1])
        col = tf.reshape(pos_flat, [-1])
        picked = tf.gather_nd(flat_idx, tf.stack([row, col], axis=1))
        topk_idx = tf.reshape(picked, [B, block_size_i, k])

        return topk_vals, topk_idx

    idx_blocks = tf.map_fn(
        process_block,
        blocks,
        dtype=tf.int32,
        parallel_iterations=1
    )  # shape: [num_blocks, B, block_size, k]

    idx_transposed = tf.transpose(idx_blocks, [1, 0, 2, 3])
    result_padded = tf.reshape(idx_transposed, [B, num_blocks * block_size, k])
    result = result_padded[:, :N, :]
    return tf.clip_by_value(result, 0, N - 1)


    # B = tf.shape(xyz)[0]
    # N = tf.shape(xyz)[1]
    # xyz_sq = tf.reduce_sum(xyz*xyz, axis=-1)  # [B,N]

    # num_blocks = tf.cast(tf.ceil(tf.cast(N, tf.float32)/block_size), tf.int32)
    # blocks     = tf.range(num_blocks)

    # def process_block(b):
    #     start = b*block_size
    #     end   = tf.minimum(start+block_size, N)
    #     xi    = xyz[:, start:end, :]                        # [B, b_i, 3]
    #     xi_sq = tf.reduce_sum(xi*xi, axis=-1)               # [B, b_i]
    #     # init top-k dists/indices
    #     top_d = tf.fill([B, tf.shape(xi)[1], k], 1e10)
    #     top_i = tf.zeros([B, tf.shape(xi)[1], k], tf.int32)

    #     num_chunks = tf.cast(tf.ceil(tf.cast(N, tf.float32)/chunk_size), tf.int32)
    #     def body(i, td, ti):
    #         s = i*chunk_size
    #         e = tf.minimum(s+chunk_size, N)
    #         xj = xyz[:, s:e, :]                             # [B, c_j, 3]
    #         dj = (xi_sq[:,:,None] + xyz_sq[:,s:e][None,:,:]
    #               -2*tf.matmul(xi, xj, transpose_b=True))   # [B, b_i, c_j]
    #         gi = tf.range(s, e)
    #         gi = tf.tile(gi[None,None,:], [B, tf.shape(xi)[1], 1])
    #         # merge top-k so far with new chunk
    #         cd = tf.concat([td, dj], axis=2)
    #         ci = tf.concat([ti, gi], axis=2)
    #         neg_cd = -cd
    #         vals, idxs = tf.nn.top_k(neg_cd, k=k)
    #         vals = -vals
    #         # gather the correct k indices
    #         flat_ci = tf.reshape(ci, [-1, tf.shape(ci)[2]])
    #         flat_id = tf.reshape(idxs, [-1, k])
    #         rows    = tf.reshape(tf.tile(tf.range(B*tf.shape(xi)[1])[:,None],[1,k]), [-1,2*k])[:,0]
    #         chosen  = tf.gather_nd(flat_ci, tf.stack([rows, tf.reshape(flat_id,[-1])],1))
    #         new_ti  = tf.reshape(chosen, [B, tf.shape(xi)[1], k])
    #         return i+1, vals, new_ti

    #     _, top_d, top_i = tf.while_loop(
    #         lambda i, *_: i < num_chunks,
    #         body,
    #         [0, top_d, top_i],
    #         shape_invariants=[
    #             tf.TensorShape([]),
    #             tf.TensorShape([None, None, None]),
    #             tf.TensorShape([None, None, None])
    #         ]
    #     )
    #     # pad block to fixed block_size so map_fn works
    #     pad = block_size - tf.shape(xi)[1]
    #     top_i = tf.concat([top_i, tf.zeros([B, pad, k], tf.int32)], axis=1)
    #     return top_i  # [B, block_size, k]

    # # run per-block KNN
    # all_blocks = tf.map_fn(process_block, blocks, dtype=tf.int32)
    # all_blocks = tf.transpose(all_blocks, [1,0,2,3])
    # # flatten & slice off padding
    # all_blocks = tf.reshape(all_blocks, [B, num_blocks*block_size, k])
    # all_blocks = all_blocks[:, :N, :]
    # return tf.clip_by_value(all_blocks, 0, N-1)



def multi_scale_group(xyz, features, neigh_k1, neigh_k2):
    """
    Perform two-scale KNN grouping: k1 and k2 neighborhoods.
    xyz: [B, N, 3], features: [B, N, C]
    neigh_k1: [B, N, k1], neigh_k2: [B, N, k2]
    returns: neigh_xyz [B, N, k1+k2, 3], neigh_feat [B, N, k1+k2, C]
    """
    B = tf.shape(xyz)[0]
    N = tf.shape(xyz)[1]
    
    # Build batch indices of shape [B, N]
    batch_idx = tf.tile(tf.reshape(tf.range(B, dtype=tf.int32), [B, 1]), [1, N])

    # === GATHER FOR K1 (Corrected) ===
    k1 = tf.shape(neigh_k1)[2]
    # Tile batch_idx to be compatible with neigh_k1's shape [B, N, k1]
    batch_idx_tiled_k1 = tf.tile(tf.expand_dims(batch_idx, axis=-1), [1, 1, k1])
    # Stack to create [batch_idx, point_idx] pairs for each neighbor
    gather_idx1 = tf.stack([batch_idx_tiled_k1, neigh_k1], axis=-1)
    
    # Gather the neighbor coordinates and features
    neigh_xyz1  = tf.gather_nd(xyz, gather_idx1)
    neigh_feat1 = tf.gather_nd(features, gather_idx1)

    # === GATHER FOR K2 (Corrected) ===
    k2 = tf.shape(neigh_k2)[2]
    # Tile batch_idx to be compatible with neigh_k2's shape [B, N, k2]
    batch_idx_tiled_k2 = tf.tile(tf.expand_dims(batch_idx, axis=-1), [1, 1, k2])
    # Stack to create [batch_idx, point_idx] pairs for each neighbor
    gather_idx2 = tf.stack([batch_idx_tiled_k2, neigh_k2], axis=-1)
    
    # Gather the neighbor coordinates and features
    neigh_xyz2  = tf.gather_nd(xyz, gather_idx2)
    neigh_feat2 = tf.gather_nd(features, gather_idx2)

    # === CONCATENATE BOTH SCALES ===
    neigh_xyz  = tf.concat([neigh_xyz1, neigh_xyz2], axis=2)
    neigh_feat = tf.concat([neigh_feat1, neigh_feat2], axis=2)

    return neigh_xyz, neigh_feat


def geometry_adaptive_scores(xyz, features, name, is_training):
    B = tf.shape(xyz)[0]
    N = tf.shape(xyz)[1]

    # 1) Figure out C_in, either static or dynamic
    C_static = features.get_shape()[-1].value
    C_in = C_static if C_static is not None else tf.shape(features)[2]

    # 2) Flatten coords and features
    flat_xyz   = tf.reshape(xyz,   [B * N, 3])           # [B*N,3]
    flat_feats = tf.reshape(features, [B * N, C_in])      # [B*N,C_in]
    # **CRUCIAL**: assert to TF that flat_feats really is [..., C_in]
    flat_feats.set_shape([None, C_in])

    # 3) Build MLP input
    concat_in  = tf.concat([flat_xyz, flat_feats], axis=1)  # [B*N, 3+C_in]
    concat_in.set_shape([None, 3 + C_in])                  # <<< tell TF the last dim

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        mlp1   = tf.layers.dense(concat_in, 64,
                                 activation=tf.nn.leaky_relu,
                                 name='gas_dense1')
        mlp2   = tf.layers.dense(mlp1,      32,
                                 activation=tf.nn.leaky_relu,
                                 name='gas_dense2')
        scores = tf.layers.dense(mlp2,      1,
                                 activation=None,
                                 name='gas_out')

    scores = tf.reshape(scores, [B, N])
    return tf.nn.softmax(scores, axis=-1)



def geometry_adaptive_sample(xyz, features, sample_ratio, name, is_training):
    """
    Sample k points out of N via 50% random + 50% top-50% learned scores.
    Returns: sampled_features [B, k, C], sampled_xyz [B, k, 3], idx_concat [B, k]
    """
    B = tf.shape(xyz)[0]
    N = tf.shape(xyz)[1]
    # Safe shape handling for channel dimension
    B = tf.shape(xyz)[0]
    N = tf.shape(xyz)[1]
    C = tf.shape(features)[2]

    # compute k = floor(N / sample_ratio)
    rf = tf.cast(sample_ratio, tf.float32)
    kf = tf.cast(N, tf.float32) / rf
    k  = tf.maximum(tf.minimum(tf.cast(tf.floor(kf), tf.int32), N), 1)

    # learned scores + random
    scores = geometry_adaptive_scores(xyz, features, '{}_GAS'.format(name), is_training)
    half_k = tf.cast(tf.floor(tf.cast(k, tf.float32) * 0.5), tf.int32)
    rand_k = k - half_k

    topk_vals, topk_idx = tf.nn.top_k(scores, half_k)
    rand_vals = tf.random_uniform([B, N], 0, 1)
    _, rand_idx = tf.nn.top_k(rand_vals, k=rand_k, sorted=False)

    idx_concat = tf.concat([topk_idx, rand_idx], axis=1)  # [B, k]

    

    # argsort returns the indices that would sort the tensor
    sort_indices = tf.contrib.framework.argsort(idx_concat, axis=1, direction='ASCENDING')
    # then gather
    idx_concat = tf.batch_gather(idx_concat, sort_indices)

            

    # gather
    batch_i   = tf.tile(tf.range(B)[:,None], [1, k])     # [B,k]
    full_idx  = tf.stack([batch_i, idx_concat], axis=2)  # [B,k,2]
    sampled_xyz = tf.gather_nd(xyz, full_idx)            # [B,k,3]

    flat_feat    = tf.reshape(features, [B * N, C])
    gather_index = tf.reshape(idx_concat + batch_i * N, [-1])
    sampled_flat = tf.gather(flat_feat, gather_index)    # [B*k, C]
    sampled_feats = tf.reshape(sampled_flat, [B, k, C])  # [B,k,C]

    return sampled_feats, sampled_xyz, idx_concat


def local_transformer_aggregation(center_xyz, center_features, neigh_xyz, neigh_feat, d_out, name, is_training):
    """
    Refactored Local Transformer Aggregation (LTA).
    Accepts pre-gathered neighbor information.
    
    :param center_xyz:      [B, N, 3]        Coordinates of the center points.
    :param center_features: [B, N, C_in]     Features of the center points.
    :param neigh_xyz:       [B, N, k, 3]     Coordinates of the neighbors.
    :param neigh_feat:      [B, N, k, C_in]  Features of the neighbors.
    :param d_out:           int              Output dimension.
    :param name:            str              Scope name.
    :param is_training:     tf.bool          Training flag.
    """
    B = tf.shape(center_xyz)[0]
    N = tf.shape(center_xyz)[1]
    k = tf.shape(neigh_xyz)[2]
    C_in = center_features.get_shape()[-1].value or tf.shape(center_features)[2]

    # ---- 1) Flatten inputs for MLP layers ----
    # Flatten center features for query and residual connection
    flat_center_feat = tf.reshape(center_features, [-1, C_in])
    # Flatten neighbor features for key and value
    flat_neigh_feat = tf.reshape(neigh_feat, [-1, C_in])
    
    # ---- 2) Positional encoding ----
    # Expand center_xyz to [B, N, 1, 3] for broadcasting
    center_xyz_expanded = tf.expand_dims(center_xyz, axis=2) 
    
    # Calculate relative position
    rel_pos = neigh_xyz - center_xyz_expanded  # [B, N, k, 3]
    rel_dist = tf.sqrt(tf.reduce_sum(tf.square(rel_pos), axis=-1, keepdims=True))
    
    # Concatenate and flatten positional encoding
    pos_enc = tf.concat([rel_dist, rel_pos], axis=-1)  # [B, N, k, 4]
    pf = tf.reshape(pos_enc, [-1, 4]) # [B*N*k, 4]
    pf.set_shape([None, 4])

    heads = 4
    head_dim = d_out // heads

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        # Positional transforms (on relative positions)
        pos_key = tf.layers.dense(pf, d_out, activation=tf.nn.leaky_relu, name='pos_key')
        pos_val = tf.layers.dense(pf, d_out, activation=None, name='pos_val')
        # Content transforms
        query = tf.layers.dense(flat_center_feat, d_out, name='query') # From center points
        key   = tf.layers.dense(flat_neigh_feat, d_out, name='key')    # From neighbor points
        val   = tf.layers.dense(flat_neigh_feat, d_out, name='value')  # From neighbor points

    # ---- 3) Split heads & compute attention ----
    # Reshape using -1 to combine Batch and Point dimensions
    key_total  = tf.reshape(key  + pos_key, [-1, k, heads, head_dim]) # Shape: [B*N, k, h, hd]
    val_total  = tf.reshape(val  + pos_val, [-1, k, heads, head_dim]) # Shape: [B*N, k, h, hd]
    query_resh = tf.reshape(query,           [-1, heads, head_dim])   # Shape: [B*N, h, hd]

    # Transpose key for dot product
    key_t = tf.transpose(key_total, [0, 2, 3, 1]) # Shape: [B*N, h, hd, k]
    q_exp = tf.expand_dims(query_resh, axis=2)    # Shape: [B*N, h, 1, hd]

    scores = tf.matmul(q_exp, key_t) / tf.sqrt(tf.cast(head_dim, tf.float32))
    scores = tf.squeeze(scores, axis=2)
    attn_w = tf.nn.softmax(scores, axis=-1)

    # ---- 4) Weighted sum, final projection ----
    attn_w_exp = tf.expand_dims(attn_w, axis=2)
    val_t = tf.transpose(val_total, [0, 2, 1, 3])
    
    weighted = tf.matmul(attn_w_exp, val_t)
    weighted = tf.squeeze(weighted, axis=2)
    
    attn_out = tf.reshape(weighted, [-1, d_out])
    
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        out_proj = tf.layers.dense(attn_out, d_out, name='out_proj')

    # ---- 5) Residual + FFN ----
    if C_in != d_out:
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            res_proj = tf.layers.dense(flat_center_feat, d_out, name='res_proj')
        combined = out_proj + res_proj
    else:
        combined = out_proj + flat_center_feat

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        ffn1 = tf.layers.dense(combined, d_out * 2, activation=tf.nn.leaky_relu, name='ffn1')
        ffn2 = tf.layers.dense(ffn1, d_out, activation=None, name='ffn2')
    ffn_out = combined + ffn2

    # ---- 6) Back to [B, N, d_out] ----
    result = tf.reshape(ffn_out, [B, N, d_out])
    result.set_shape([None, None, d_out])
    return result

class Network:
    """
    RandLA-Net++ architecture with:
      1) Geometry-Adaptive Sampling (GAS)
      2) Multi-Scale Local Grouping (MSLG)
      3) Local Transformer Aggregation (LTA)
    """

    def __init__(self, dataset, config):
        """
        dataset: Semantic3D instance (provides .flat_inputs, .train_init_op, .val_init_op, etc.)
        config: ConfigSemantic3D with fields:
            num_layers, d_out, sub_sampling_ratio, k_n, num_points, num_features,
            num_classes, ignored_label_inds, learning_rate, lr_decays,
            train_steps, val_steps, train_sum_dir, saving, saving_path, etc.
        """
        self.config = config
        L = config.num_layers

        # ================== Placeholders & Dataset Inputs ==================
        flat_inputs = dataset.flat_inputs
        assert len(flat_inputs) == 5 * L + 4, "Expected {} flat inputs, got {}".format(5*L+4,len(flat_inputs))

        self.inputs = {
    'xyz':        flat_inputs[0: L],         # Correct: Provided by 'pts_list'
    'neigh_idx':  flat_inputs[L: 2*L],       # Correct: Provided by 'neigh_list'
    'sub_idx':    flat_inputs[2*L: 3*L],       # Correct: Provided by 'sub_list'
    'interp_idx': flat_inputs[4*L: 5*L],     # CORRECTED: Your 'up_list' is here
    'features':   flat_inputs[5*L],          # CORRECTED: Your 'batch_feat' is here
    'labels':     flat_inputs[5*L + 1],      # CORRECTED: Your 'batch_lbl' is here
    'input_inds': flat_inputs[5*L + 2],      # CORRECTED: Your 'batch_pc_idx' is here
    'cloud_inds': flat_inputs[5*L + 3],      # CORRECTED: Your 'batch_cloud_idx' is here
}
        # Explicitly set static shape for features
        self.inputs['features'].set_shape([None, None, config.num_features])

        self.encoder_xyz    = [self.inputs['xyz'][0]]
        self.sample_indices = []
        self.encoder_neigh_idx = []

        # Training controls
        self.is_training   = tf.placeholder(tf.bool, shape=(), name='is_training')
        self.training_step = 1
        self.training_epoch= 0
        self.mIou_list     = [0.0]
        self.class_weights = DP.get_class_weights(dataset.name)

        # ================== Build Inference Graph ==================
        with tf.variable_scope('inference'):
            self.logits = self.inference(self.inputs, self.is_training)
            # [B, N, num_classes]

        # ================== Loss, Optimizer, Results ==================
        with tf.variable_scope('loss'):
            self._build_loss()

        with tf.variable_scope('optimizer'):
            self._build_optimizer()

        with tf.variable_scope('results'):
            self._build_results()

        # ================== Session & Saver ==================
        self._build_session()

    def _validate_input_shapes(self, xyz_tensor, features_tensor):
        """Adds a runtime assertion to ensure xyz and features have matching dimensions."""
        shape_xyz = tf.shape(xyz_tensor)
        shape_feat = tf.shape(features_tensor)
        
        assert_op = tf.Assert(
            tf.equal(shape_xyz[1], shape_feat[1]), # Check that Num_Points are equal
            [
                "FATAL: Input data shape mismatch.",
                "The number of points in the 'xyz' tensor does not match the 'features' tensor.",
                "XYZ shape:", shape_xyz,
                "Features shape:", shape_feat
            ]
        )
        # This dependency ensures the assertion is checked before the tensor is used.
        with tf.control_dependencies([assert_op]):
            return tf.identity(features_tensor)

    def inference(self, inputs, is_training):
        """
        Encoder‐Decoder with:
          • Initial projection: features→8 dims
          • For each layer i:
              - MSLG (k1, k2)
              - LTA → d_out[i]
              - GAS → subsample to next layer
          • Bottleneck MLP
          • Decoder: nearest upsample + skip MLP + optional LTA
          • Final classification MLP
        """
        d_out     = self.config.d_out             # e.g. [16,64,128,256,512]
        sub_ratio = self.config.sub_sampling_ratio# e.g. [4,4,4,4,2]
        L         = self.config.num_layers
        k1        = self.config.k_n               # e.g. 16
        k2        = k1 * 2                        # e.g. 32

        xyz_input_shape = tf.shape(inputs['xyz'][0])
        features_input_shape = tf.shape(inputs['features'])
        
        

        feat0 = self._validate_input_shapes(inputs['xyz'][0], inputs['features'])
    

        # -------- Initial Projection -----------
        # feat0 = tf.cast(inputs['features'], tf.float32)   # [B, N, num_features]
        feat0 = tf.cast(feat0, tf.float32)
        
        C0 = self.config.num_features
        with tf.variable_scope('fc0', reuse=tf.AUTO_REUSE):
            f0_flat = tf.reshape(feat0, [-1, C0])
            f0_dense = tf.layers.dense(f0_flat, 8, activation=None, name='fc0_dense')
            eps = tf.convert_to_tensor(1e-6, dtype=tf.float32, name='fc0_bn_eps')
            f0_bn = tf.layers.batch_normalization(f0_dense, axis=-1, momentum=0.99, epsilon=eps, training=is_training, name='fc0_bn')
            f0_act = tf.nn.leaky_relu(f0_bn)
            feature = tf.reshape(f0_act, [-1, tf.shape(feat0)[1], 8])
            feature.set_shape([None, None, 8])

        f_enc_list = [feature]  # store features per layer

        # ===================== Encoder =====================
        for i in range(L):
            xyz_i = self.encoder_xyz[i]   # [B, N_i, 3]
            feat_i= f_enc_list[i]         # [B, N_i, C_i]
            
            # Ensure static shape is available
            if feat_i.get_shape().ndims is not None and feat_i.get_shape()[-1].value is None:
                # Explicitly set channel dimension if missing
                feat_i = tf.reshape(feat_i, [tf.shape(feat_i)[0], tf.shape(feat_i)[1], -1])
                feat_i.set_shape([None, None, feat_i.get_shape()[-1].value or 8 if i==0 else d_out[i-1]])

            # Debugging: Print current layer info
            # xyz_i = debug_print(xyz_i, "[Encoder Layer {}] XYZ shape".format(i))
            # feat_i = debug_print(feat_i, "[Encoder Layer {}] Features shape".format(i))

            # a) Multi-scale grouping
            neigh_idx_k1 = knn_blocked(xyz_i, k1, block_size=4096)
            neigh_idx_k2 = knn_blocked(xyz_i, k2, block_size=4096)
            
            # Debugging: Print neighbor indices
            # neigh_idx_k1 = debug_print(neigh_idx_k1, "[Encoder Layer {}] K1 neighbor indices".format(i))
            # neigh_idx_k2 = debug_print(neigh_idx_k2, "[Encoder Layer {}] K2 neighbor indices".format(i))

            neigh_xyz, neigh_feat = multi_scale_group(xyz_i, feat_i, neigh_idx_k1, neigh_idx_k2)

            self.encoder_neigh_idx.append(neigh_idx_k1)

            # b) Local Transformer Aggregation → [B, N_i, d_out[i]]
            with tf.variable_scope('Enc_LTA_{}'.format(i), reuse=tf.AUTO_REUSE):
                feat_ta = local_transformer_aggregation(
                center_xyz=xyz_i, 
                center_features=feat_i, 
                neigh_xyz=neigh_xyz,
                neigh_feat=neigh_feat,
                d_out=d_out[i], 
                name='Enc_LTA_{}'.format(i), 
                is_training=is_training
            )
                # feat_ta = debug_print(feat_ta, "[Encoder LTA {}] Output shape".format(i))

            # c) MLP projection
            feat_ta_flat = tf.reshape(feat_ta, [-1, d_out[i]])  # [B*N_i, d_out[i]]
            with tf.variable_scope('Enc_MLP_{}'.format(i), reuse=tf.AUTO_REUSE):
                mlp1 = tf.layers.dense(feat_ta_flat, d_out[i], activation=None, name='mlp_dense')
                bn1  = tf.layers.batch_normalization(mlp1, axis=-1, momentum=0.99, epsilon=1e-6, training=is_training, name='mlp_bn')
                feat_enc = tf.nn.leaky_relu(bn1)
            feat_enc = tf.reshape(feat_enc, [-1, tf.shape(feat_ta)[1], d_out[i]])
            feat_enc.set_shape([None, None, d_out[i]])

            sampled_feat, sampled_xyz, sample_idx = geometry_adaptive_sample(
                xyz_i, feat_enc, sub_ratio[i], 'GAS_{}'.format(i), is_training
            )
            sampled_feat.set_shape([None, None, d_out[i]])
            
            self.encoder_xyz.append(sampled_xyz)
            self.sample_indices.append(sample_idx)
            f_enc_list.append(sampled_feat)
            # self.encoder_neigh_idx.append(neigh_idx_k1)

        # ===================== Bottleneck =====================
        feat_bot = f_enc_list[-1]  # [B, N_L, d_out[L-1]]
        # Set static shape if missing
        if feat_bot.get_shape().ndims is not None and feat_bot.get_shape()[-1].value is None:
            feat_bot = tf.reshape(feat_bot, [tf.shape(feat_bot)[0], tf.shape(feat_bot)[1], d_out[-1]])
            feat_bot.set_shape([None, None, d_out[-1]])
            
        with tf.variable_scope('Bottleneck', reuse=tf.AUTO_REUSE):
            bot_flat = tf.reshape(feat_bot, [-1, d_out[-1]])  # [B*N_L, C_bot]
            bot_dense= tf.layers.dense(bot_flat, d_out[-1], activation=None, name='bot_dense')
            bot_bn   = tf.layers.batch_normalization(bot_dense, axis=-1, momentum=0.99, epsilon=1e-6,
                                                     training=is_training, name='bot_bn')
            feat_bot_out = tf.nn.leaky_relu(bot_bn)       # [B*N_L, C_bot]
        feat_bot_out = tf.reshape(feat_bot_out, [-1, tf.shape(feat_bot)[1], d_out[-1]])  # [B, N_L, C_bot]
        feat_bot_out.set_shape([None, None, d_out[-1]])  # Set static shape
        # feat_bot_out = debug_print(feat_bot_out, "[Bottleneck] Output shape")

        # ===================== Decoder =====================
        f_dec = feat_bot_out  # [B, N_L, C_bot]
        for j in range(L):
            i = L - j - 1
            skip_feat = f_enc_list[i]
            skip_xyz = self.encoder_xyz[i]    # [B, N_i, 3]
            skip_neigh_idx_k1 = self.encoder_neigh_idx[i]
            neigh_idx_k1 = self.encoder_neigh_idx[i]
            # Fetch precomputed 1-NN indices for upsampling
            interp_idx_dec = inputs['interp_idx'][i]  # [B, N_i, 1]
            
            # Ensure static shapes
            if skip_feat.get_shape().ndims is not None and skip_feat.get_shape()[-1].value is None:
                skip_feat = tf.reshape(skip_feat, [tf.shape(skip_feat)[0], tf.shape(skip_feat)[1], d_out[i]])
                skip_feat.set_shape([None, None, d_out[i]])
            
            # Debugging: Print decoder info
            # f_dec = debug_print(f_dec, "[Decoder Layer {}] Input features shape".format(j))
            # skip_feat = debug_print(skip_feat, "[Decoder Layer {}] Skip features shape".format(j))
            # skip_xyz = debug_print(skip_xyz, "[Decoder Layer {}] Skip XYZ shape".format(j))
            # interp_idx_dec = debug_print(interp_idx_dec, "[Decoder Layer {}] Interp indices".format(j))

            # Nearest neighbor interpolation
            # reshape f_dec: [B, N_{i+1}, C]
            B_dec = tf.shape(f_dec)[0]
            N_fine = tf.shape(skip_feat)[1]
            
            # Set static shape for f_dec if missing
            if f_dec.get_shape().ndims is not None and f_dec.get_shape()[-1].value is None:
                f_dec = tf.reshape(f_dec, [tf.shape(f_dec)[0], tf.shape(f_dec)[1], d_out[i+1] if j>0 else d_out[-1]])
                f_dec.set_shape([None, None, d_out[i+1] if j>0 else d_out[-1]]) 
            
            # flatten and gather
            flat_fdec = tf.reshape(f_dec, [-1, d_out[i+1] if j>0 else d_out[-1]])
            batch_offset_dec = tf.reshape(tf.range(B_dec, dtype=tf.int32) * tf.shape(f_dec)[1], [B_dec, 1, 1])
            full_indices = tf.reshape(interp_idx_dec + batch_offset_dec, [-1, 1])
            up_feat_flat = tf.gather(flat_fdec, full_indices[:,0])
            up_feat = tf.reshape(up_feat_flat, [B_dec, N_fine, d_out[i+1] if j>0 else d_out[-1]])
            up_feat.set_shape([None, None, d_out[i+1] if j>0 else d_out[-1]])  # Set static shape
            
            # up_feat = debug_print(up_feat, "[Decoder Layer {}] Up-sampled features shape".format(j))

            # b) Concatenate skip & up
            cat_feat = tf.concat([skip_feat, up_feat], axis=-1)
            with tf.variable_scope('Dec_MLP_{}'.format(j), reuse=tf.AUTO_REUSE):
                # ... (MLP code remains the same) ...
                C_cat = cat_feat.get_shape()[-1].value or (d_out[i] + (d_out[i+1] if j > 0 else d_out[-1]))
                cat_flat = tf.reshape(cat_feat, [-1, C_cat])
                mlp_dec = tf.layers.dense(cat_flat, d_out[i], activation=None, name='dec_dense')
                bn_dec  = tf.layers.batch_normalization(mlp_dec, axis=-1, momentum=0.99, epsilon=1e-6, training=is_training, name='dec_bn')
                feat_dec = tf.nn.leaky_relu(bn_dec)
            feat_dec = tf.reshape(feat_dec, [B_dec, N_fine, d_out[i]])
            feat_dec.set_shape([None, None, d_out[i]])  # Set static shape
            # feat_dec = debug_print(feat_dec, "[Decoder Layer {}] MLP output shape".format(j))

            # c) Optional LTA on skip for boundary refinement
            with tf.variable_scope('Dec_LTA_{}'.format(j), reuse=tf.AUTO_REUSE):
            # Gather neighbor information for the decoder LTA
            # Note: We use the decoder's features (feat_dec) but the encoder's geometry (skip_xyz and its neighbors)
                dec_neigh_xyz, dec_neigh_feat = multi_scale_group(skip_xyz, feat_dec, skip_neigh_idx_k1, skip_neigh_idx_k1)

                # Call the refactored LTA block
                feat_dec = local_transformer_aggregation(
                    center_xyz=skip_xyz, 
                    center_features=feat_dec, 
                    neigh_xyz=dec_neigh_xyz,
                    neigh_feat=dec_neigh_feat,
                    d_out=d_out[i],
                    name='Dec_LTA_{}'.format(j), 
                    is_training=is_training
                )

            f_dec = feat_dec  # propagate up

        # ===================== Final Classification =====================
        final_feat = f_dec  # [B, N, d_out[0]]
        # Set static shape if missing
        if final_feat.get_shape().ndims is not None and final_feat.get_shape()[-1].value is None:
            final_feat = tf.reshape(final_feat, [tf.shape(final_feat)[0], tf.shape(final_feat)[1], d_out[0]])
            final_feat.set_shape([None, None, d_out[0]])
            
        with tf.variable_scope('Classifier', reuse=tf.AUTO_REUSE):
            final_flat = tf.reshape(final_feat, [-1, d_out[0]])  # [B*N, d_out[0]]
            fc1 = tf.layers.dense(final_flat,   64, activation=None, name='fc1_dense')
            bn1 = tf.layers.batch_normalization(fc1, axis=-1, momentum=0.99, epsilon=1e-6,
                                                training=is_training, name='fc1_bn')
            a1  = tf.nn.leaky_relu(bn1)
            fc2 = tf.layers.dense(a1,  32, activation=None, name='fc2_dense')
            bn2 = tf.layers.batch_normalization(fc2, axis=-1, momentum=0.99, epsilon=1e-6,
                                                training=is_training, name='fc2_bn')
            a2  = tf.nn.leaky_relu(bn2)
            drop1 = tf.layers.dropout(a2, rate=0.5, training=is_training, name='dropout')
            fc3   = tf.layers.dense(drop1, self.config.num_classes, activation=None, name='fc_out')
        logits = tf.reshape(fc3, [-1, tf.shape(final_feat)[1], self.config.num_classes])  # [B, N, num_classes]
        
        # logits = debug_print(logits, "[Classifier] Final logits shape")
        return logits

    def _build_loss(self):
        """
        Build weighted cross-entropy loss, ignoring specified labels.
        """
        num_classes = self.config.num_classes
        flat_logits = tf.reshape(self.logits, [-1, num_classes])  # [B*N, C]
        flat_labels = tf.reshape(self.inputs['labels'], [-1])     # [B*N]
        
        # Create mask for valid labels (not ignored)
        mask = tf.ones_like(flat_labels, dtype=tf.bool)
        for ign in self.config.ignored_label_inds:
            mask = tf.logical_and(mask, tf.not_equal(flat_labels, ign))
        
        valid_idx = tf.where(mask)
        valid_logits = tf.gather(flat_logits, valid_idx)  # [num_valid, C]
        valid_labels = tf.gather(flat_labels, valid_idx)  # [num_valid]
        
        valid_logits = tf.squeeze(valid_logits, axis=1)      # [num_valid, C]
        valid_labels = tf.squeeze(valid_labels, axis=1) 

        # Ensure labels are in valid range [0, num_classes-1]
        valid_labels = tf.clip_by_value(valid_labels, 0, num_classes-1)
        
        with tf.variable_scope('loss_weights', reuse=tf.AUTO_REUSE):
            cw = tf.convert_to_tensor(self.class_weights, dtype=tf.float32)  # [C]
            one_hot = tf.one_hot(valid_labels, depth=num_classes)  # [num_valid, C]
            weights = tf.reduce_sum(cw * one_hot, axis=1)          # [num_valid]
            
            unweighted = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=valid_logits, 
                labels=one_hot
            )
            weighted = unweighted * weights
            self.loss = tf.reduce_mean(weighted)

    def _build_optimizer(self):
        """
        Build Adam optimizer and ensure batchnorm update ops execute.
        """
        self.learning_rate = tf.Variable(self.config.learning_rate, trainable=False, name='learning_rate')
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.loss)

    def _build_results(self):
        num_classes = self.config.num_classes
        flat_logits = tf.reshape(self.logits, [-1, num_classes])
        flat_labels = tf.reshape(self.inputs['labels'], [-1])
        
        # Create mask for valid labels
        mask = tf.ones_like(flat_labels, dtype=tf.bool)
        for ign in self.config.ignored_label_inds:
            mask = tf.logical_and(mask, tf.not_equal(flat_labels, ign))
        
        valid_idx = tf.where(mask)
        valid_logits = tf.gather(flat_logits, valid_idx)
        valid_labels = tf.gather(flat_labels, valid_idx)
        
        valid_logits = tf.squeeze(valid_logits, axis=1)      # [num_valid, C]
        valid_labels = tf.squeeze(valid_labels, axis=1) 
        # Ensure labels are in valid range
        valid_labels = tf.clip_by_value(valid_labels, 0, num_classes-1)
        
        self.correct_prediction = tf.nn.in_top_k(valid_logits, valid_labels, 1)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        self.prob_logits = tf.nn.softmax(flat_logits)

        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.scalar('learning_rate', self.learning_rate)

    def _build_session(self):
        """
        Configure TensorFlow session, saver, and summary writer.
        """
        all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.saver = tf.train.Saver(all_vars, max_to_keep=100)
        sess_cfg = tf.ConfigProto()
        sess_cfg.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_cfg)
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.config.train_sum_dir, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def train(self, dataset):
        """
        Train loop (similar to original RandLA-Net), but now using our novel architecture.
        """
        def log_out(msg, log_file):
            print(msg)
            log_file.write(msg + '\n')
            log_file.flush()

        log_file = open('log_train_sem3d.txt', 'a')
        log_out("****EPOCH {}****".format(self.training_epoch), log_file)

        self.sess.run(dataset.train_init_op)
        step_times = []

        while self.training_epoch < self.config.max_epoch:
            t_start = time.time()
            try:
                ops = [self.train_op, self.merged, self.loss, self.accuracy]
                _, summary, l_out, acc_b = self.sess.run(ops, {self.is_training: True})
                self.train_writer.add_summary(summary, self.training_step)

                t_end = time.time()
                step_time = (t_end - t_start) * 1000
                step_times.append(step_time)

                if self.training_step % 20 == 0:
                    recent = step_times[-20:] if len(step_times) >= 20 else step_times
                    avg_time = np.mean(recent)
                    log_out(
                        "Step {step:08d}  L_out={loss:.3f}  Acc={acc:.2f}  --- {time:8.2f} ms/batch".format(
                            step=self.training_step,
                            loss=l_out,
                            acc=acc_b,
                            time=avg_time
                        ),
                        log_file
                    )

                self.training_step += 1

            except tf.errors.OutOfRangeError:
                # End of epoch → evaluate
                m_iou = self.evaluate(dataset)
                if m_iou > max(self.mIou_list):
                    snap_dir = join(self.config.saving_path, 'snapshots')
                    if not exists(snap_dir):
                        makedirs(snap_dir)
                    self.saver.save(self.sess, join(snap_dir, 'snap'), global_step=self.training_step)
                self.mIou_list.append(m_iou)
                log_out("Best m_IoU is: {:.3f}".format(max(self.mIou_list)), log_file)

                self.training_epoch += 1
                self.sess.run(dataset.train_init_op)

                # Decay LR
                lr_new = self.config.lr_decays.get(self.training_epoch, 1.0)
                self.sess.run(self.learning_rate.assign(self.learning_rate * lr_new))
                log_out("****EPOCH {}****".format(self.training_epoch), log_file)

            except tf.errors.InvalidArgumentError as e:
                log_out("Numerical error at step {}".format(self.training_step), log_file)
                log_out("Error details: {}".format(str(e)), log_file)

                crash_dir = join(self.config.saving_path, 'crash_snapshot')
                if not exists(crash_dir):
                    makedirs(crash_dir)
                self.saver.save(self.sess, crash_dir)
                log_out("Saved crash snapshot to: {}".format(crash_dir), log_file)

                break

        log_out("Training finished", log_file)
        log_file.close()
        self.sess.close()


    def evaluate(self, dataset):
        """
        Validation loop returning mean IoU (%) for Semantic3D.
        """
        def log_out(msg, log_file):
            print(msg)
            log_file.write(msg + '\n')
            log_file.flush()

        log_file = open('log_eval_sem3d.txt', 'a')
        self.sess.run(dataset.val_init_op)

        num_classes = self.config.num_classes
        gt_classes = np.zeros(num_classes, dtype=np.int64)
        pos_classes= np.zeros(num_classes, dtype=np.int64)
        tp_classes = np.zeros(num_classes, dtype=np.int64)
        total_correct = 0
        total_seen = 0

        for step in range(self.config.val_steps):
            if step % 50 == 0:
                log_out("{} / {}".format(step,self.config.val_steps), log_file)
            try:
                probs, labels = self.sess.run([self.prob_logits, self.inputs['labels']], {self.is_training: False})
                # probs: [B*N, C], labels: [B, N]
                B = labels.shape[0]
                flat_labels = labels.reshape(-1)                # [B*N]
                flat_preds  = np.argmax(probs, axis=1).astype(np.int32)  # [B*N]

                if self.config.ignored_label_inds:
                    mask = np.ones_like(flat_labels, dtype=bool)
                    for ign in self.config.ignored_label_inds:
                        mask &= (flat_labels != ign)
                    valid_labels = flat_labels[mask]
                    valid_preds  = flat_preds[mask]
                else:
                    valid_labels = flat_labels
                    valid_preds  = flat_preds

                total_correct += np.sum(valid_labels == valid_preds)
                total_seen    += valid_labels.shape[0]

                conf = confusion_matrix(valid_labels, valid_preds, labels=np.arange(num_classes))
                gt_classes += np.sum(conf, axis=1)
                pos_classes+= np.sum(conf, axis=0)
                tp_classes += np.diagonal(conf)

            except tf.errors.OutOfRangeError:
                break

        iou_list = []
        for c in range(num_classes):
            denom = gt_classes[c] + pos_classes[c] - tp_classes[c]
            iou = tp_classes[c] / float(denom) if denom > 0 else 0.0
            iou_list.append(iou)
        mean_iou = np.mean(iou_list)
        mean_iou_pct = 100.0 * mean_iou

        acc_val = total_correct / float(total_seen) if total_seen > 0 else 0.0
        log_out("Evaluation accuracy: {:.4f}".format(acc_val), log_file)
        log_out("Mean IoU: {:.4f}".format(mean_iou), log_file)
        log_out("Mean IoU = {:.1f}%".format(mean_iou_pct), log_file)
        s = "{:.2f} | ".format(mean_iou_pct) + " ".join(["{:.2f}".format(100 * x) for x in iou_list])
        log_out("-" * len(s), log_file)
        log_out(s, log_file)
        log_out("-" * len(s) + "\n", log_file)

        log_file.close()
        return mean_iou_pct
