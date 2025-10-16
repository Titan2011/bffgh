from helper_tool import ConfigSemantic3D as cfg
from helper_tool import DataProcessing as DP
from RandLANet import Network, knn_blocked
from dataset import Semantic3D
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os
import numpy as np
import logging
from collections import defaultdict
from os.path import join
from tester_Semantic3D import ModelTester
from boundary_metrics import BoundaryAwareMetrics, CurvatureAwareEvaluator
from ablation_study import AblationStudy
from visualization_tools import BoundaryVisualization
import wandb  # For experiment tracking (optional but recommended for A* publications)
import itertools

def get_dataloader(mode):
    # NOTE: Your original code imports 'Semantic3D' from 'dataset.py' but provides
    # an implementation for 'Semantic3DDataset'. Ensure the correct class is instantiated.
    # Assuming 'Semantic3D' is the intended class from your import.
    dataset = Semantic3D(mode)
    
    # Ensure a valid dataset was created
    if len(dataset) == 0:
        logging.warning(f"DataLoader for mode '{mode}' has 0 samples. Check your dataset path and splits.")
        # Return an empty loader to prevent crashes
        return DataLoader(dataset, batch_size=cfg.batch_size), dataset

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size if mode == 'training' else cfg.val_batch_size,
        shuffle=(mode == 'training'), # Only shuffle training data
        num_workers=4,
        collate_fn=dataset.collate_fn # Use the static collate function
    )
    return dataloader, dataset

def train(FLAGS):
    # Initialize experiment tracking (optional but recommended)
    # wandb.init(project="gas-3d-segmentation", config=cfg.__dict__)

    train_loader, _ = get_dataloader('training')
    val_loader, _ = get_dataloader('validation')

    device = torch.device(f"cuda:{FLAGS.gpu}" if torch.cuda.is_available() else "cpu")
    model = Network(cfg)
    model.to(device)

    print("DEBUG A | device:", device)
    print("DEBUG A | param device:", next(model.parameters()).device)
    print("DEBUG A | model.training (should be True):", model.training)
    print("DEBUG A | any param.requires_grad:", any(p.requires_grad for p in model.parameters()))

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=cfg.lr_decays.get(0, 0.95))

    # Get class weights for loss function
    class_weights = torch.from_numpy(DP.get_class_weights('Semantic3D').squeeze()).float().to(device)

    # --- FIX: Set a safe ignore index for the loss function ---
    # Use a value outside the range of possible class indices [0, num_classes-1]
    ignored_label_index = 255
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignored_label_index)

    # Initialize metrics
    boundary_metrics = BoundaryAwareMetrics(cfg.num_classes)
    curvature_evaluator = CurvatureAwareEvaluator(cfg.num_classes)
    visualizer = BoundaryVisualization()

    best_boundary_miou = 0.0

    # Create an iterator for the training dataloader
    train_loader_iter = iter(train_loader)

    for epoch in range(cfg.max_epoch):
        model.train()
        train_loss = 0.0

        # Wrap the step loop with tqdm for a progress bar
        pbar = tqdm(range(cfg.train_steps), desc=f"Epoch {epoch+1}/{cfg.max_epoch}", ncols=100)
        for i in pbar:
            try:
                # Get the next batch from the iterator
                batch = next(train_loader_iter)
            except StopIteration:
                # If the iterator is exhausted, re-initialize it to loop over the data again
                train_loader_iter = iter(train_loader)
                batch = next(train_loader_iter)
            
            # Move data to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            print(f"DEBUG B | iter {i} | batch keys: {list(batch.keys())}")
            try:
                print("DEBUG B | shapes xyz, features, labels:", batch['xyz'].shape, batch['features'].shape, batch['labels'].shape)
            except Exception as e:
                print("DEBUG B | shape error:", e)
            print("DEBUG B | sample label uniques (orig):", torch.unique(batch['labels']))

            optimizer.zero_grad()

            # Forward pass
            end_points = model(batch)

            print("DEBUG C1 | logits present keys:", 'logits' in end_points)
            if 'logits' in end_points:
                print("DEBUG C1 | logits.shape:", end_points['logits'].shape)

            logits = end_points['logits']  # [B, N, C]
            labels = batch['labels']      # [B, N], contains original labels [0-8]

            # --- FIX: Remap labels to be compatible with CrossEntropyLoss ---
            remapped_labels = labels - 1
            remapped_labels[labels == 0] = ignored_label_index

            # Flatten logits and remapped labels for loss calculation
            logits = logits.view(-1, cfg.num_classes)
            labels = remapped_labels.view(-1)

            print("DEBUG C2 | loss input shapes:", logits.shape, labels.shape)
            loss = loss_fn(logits, labels)
            print("DEBUG C2 | loss:", loss.item())

            p0 = next(model.parameters()).detach().cpu().clone()

            loss.backward()

            # ----------------- DEBUG D (after backward, before step) -----------------
            n_grads = sum(1 for p in model.parameters() if p.grad is not None)
            max_grad = max((p.grad.abs().max().item() for p in model.parameters() if p.grad is not None), default=0.)
            print("DEBUG D | n params w/ grad:", n_grads, "max grad:", max_grad)

            optimizer.step()

            p1 = next(model.parameters()).detach().cpu()
            print("DEBUG E | max param change:", (p1 - p0).abs().max().item())

            train_loss += loss.item()
            
            # Update the progress bar postfix with the current loss
            pbar.set_postfix(loss=loss.item())

        scheduler.step()
        
        avg_train_loss = train_loss / cfg.train_steps
        print(f"\nEpoch {epoch+1} Average Training Loss: {avg_train_loss:.4f}")


        # Validation with boundary-aware metrics
        if (epoch + 1) % 1 == 0:  # Validate every 5 epochs
            model.eval()
            boundary_metrics.reset()
            # curvature_evaluator.reset()
            
            all_curvature_results = defaultdict(list)
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                           batch[k] = v.to(device)
                    
                    outputs = model(batch)
                    
                    # --- FIX: Pass the knn_blocked function to the metrics ---
                    metrics_knn_fn = lambda q, s, k: knn_blocked(q, s, k)

                    # Update stateful boundary metrics
                    boundary_metrics.update(outputs['logits'], batch['labels'], batch['xyz'])
                    
                    # --- FIX: Evaluate stateless curvature metrics and accumulate results ---
                    # The CurvatureAwareEvaluator does not have a .reset() method, so we accumulate results.
                    remapped_labels_val = batch['labels'].clone() - 1
                    remapped_labels_val[batch['labels'] == 0] = ignored_label_index
                    
                    curvature_results = curvature_evaluator.evaluate(
                        outputs['logits'], remapped_labels_val, batch['xyz'], knn_fn=metrics_knn_fn
                    )
                    for key, value in curvature_results.items():
                        all_curvature_results[key].append(value)

                    # Update metrics
                    # boundary_metrics.update(predictions, batch['labels'], batch['xyz'], knn_fn=knn_fn)
                    # curvature_metrics = curvature_evaluator.evaluate(predictions, batch['labels'], batch['xyz'], knn_fn=knn_fn, k=32)

            # Compute metrics
            # --- Finalize and aggregate metrics ---
            # Pass the knn_fn again for the final computation step
            final_metrics = boundary_metrics.compute_metrics(knn_fn=metrics_knn_fn, boundary_widths=[0.05, 0.1, 0.2])
            
            # Average the accumulated curvature results
            for key, values in all_curvature_results.items():
                final_metrics[key] = np.nanmean([v for v in values if not np.isnan(v)])

            logging.info(f"\n--- Validation Epoch {epoch+1} ---")
            logging.info(f"  Overall mIoU: {final_metrics.get('mIoU', 0.0):.4f}")
            
            # Log metrics for all specified boundary widths
            for bw in [0.05, 0.1, 0.2]:
                b_miou_key = f'boundary_mIoU_w{bw:.3f}'
                b_miou = final_metrics.get(b_miou_key, 0.0)
                logging.info(f"  Boundary mIoU (w={bw:.2f}m): {b_miou:.4f}")

            # Use a primary boundary width for saving the best model
            primary_b_miou = final_metrics.get('boundary_mIoU_w0.100', 0.0)
            if primary_b_miou > best_boundary_miou:
                best_boundary_miou = primary_b_miou
                logging.info(f"*** New best model! Saving... Best Boundary mIoU: {best_boundary_miou:.4f} ***")
                log_dir = getattr(cfg, 'log_dir', 'logs')
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                save_path = os.path.join(log_dir, 'semantic3d_best_model.pth')
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'metrics': final_metrics}, save_path)
            # metrics = boundary_metrics.compute_metrics()
            # metrics = boundary_metrics.compute_metrics(knn_fn=knn_fn, boundary_widths=[0.05, 0.10, 0.20], k_boundary=32)

            # print(f"\nEpoch {epoch+1} Validation Results:")
            # print(f"Overall mIoU: {metrics['mIoU']:.4f}")
            # print(f"Boundary mIoU: {metrics['boundary_mIoU']:.4f}")
            # print(f"Boundary mF1: {metrics['boundary_mF1']:.4f}")
            # print(f"Boundary Improvement Ratio: {metrics.get('boundary_improvement_ratio', 0):.4f}")

            # # Save best model based on boundary mIoU
            # if metrics['boundary_mIoU'] > best_boundary_miou:
            #     best_boundary_miou = metrics['boundary_mIoU']
            #     torch.save({
            #         'epoch': epoch,
            #         'model_state_dict': model.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict(),
            #         'metrics': metrics
            #     }, f'best_model_boundary_miou_{best_boundary_miou:.4f}.pth')
            #     print(f"Saved best model with boundary mIoU: {best_boundary_miou:.4f}")

            # Log to wandb if using
            # wandb.log(metrics)

        # Periodic visualization
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                # Get one batch for visualization
                batch = next(iter(val_loader))
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)

                # Enable auxiliary info storage
                if hasattr(model, 'store_aux_info'):
                    model.store_aux_info = True
                    model.aux_info = []

                end_points = model(batch)
                predictions = end_points['logits'].argmax(dim=-1)

                # Visualize first sample in batch
                visualizer.visualize_boundary_errors(
                    batch['xyz'][0].cpu(),
                    predictions[0].cpu(),
                    batch['labels'][0].cpu(),
                    f'epoch_{epoch+1}',
                    cfg.num_classes
                )

                # Visualize sampling distribution for first layer
                if hasattr(model, 'aux_info') and model.aux_info:
                    visualizer.visualize_sampling_distribution(
                        batch['xyz'][0].cpu(),
                        model.aux_info[0]['learned_indices'][0].cpu(),
                        {k: v[0].cpu() for k, v in model.aux_info[0].items()},
                        f'epoch_{epoch+1}',
                        0
                    )
                
                if hasattr(model, 'store_aux_info'):
                    model.store_aux_info = False

#####################################################################################
######################### PyTorch Dataset Implementation ############################
#####################################################################################

class Semantic3DDataset(torch.utils.data.Dataset):
    def __init__(self, mode, config):
        self.mode = mode
        self.config = config
        self.sub_pc_folder = join(config.data_path, 'input_{:.3f}'.format(config.sub_grid_size))

        # Use the train/val split from the original RandLA-Net paper for consistency
        self.all_files = np.sort([join(self.sub_pc_folder, f) for f in os.listdir(self.sub_pc_folder) if f.endswith('.ply')])
        # This mapping is specific to the reduced benchmark of Semantic3D
        self.all_splits = [0, 1, 4, 5, 3, 4, 3, 0, 1, 2, 3, 4, 2, 0, 5]
        self.val_split_id = 1

        self.files = []
        if self.mode == 'training':
            self.files = [f for i, f in enumerate(self.all_files) if self.all_splits[i] != self.val_split_id]
        elif self.mode in ['validation', 'test']:
            self.files = [f for i, f in enumerate(self.all_files) if self.all_splits[i] == self.val_split_id]
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        logging.info(f"Found {len(self.files)} files for {mode}ing.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        """Load a .ply file (ASCII) and return a dict with points and optional labels.

        Returns:
            dict with keys:
                'points' : numpy array of shape (N,3)
                'labels' : numpy array of shape (N,) or None
                'file_path' : original file path
        """
        file_path = self.files[index]

        # Basic ASCII PLY reader robust for common Semantic3D files
        with open(file_path, 'r') as f:
            header_lines = []
            line = f.readline()
            if not line.startswith('ply'):
                raise ValueError(f"Unsupported or corrupted PLY file: {file_path}")
            header_lines.append(line)
            vertex_count = None
            properties = []
            while True:
                line = f.readline()
                if not line:
                    raise ValueError(f"Unexpected end of file while reading header: {file_path}")
                header_lines.append(line)
                line_strip = line.strip()
                if line_strip.startswith('element vertex'):
                    parts = line_strip.split()
                    if len(parts) >= 3:
                        try:
                            vertex_count = int(parts[2])
                        except ValueError:
                            vertex_count = None
                elif line_strip.startswith('property'):
                    parts = line_strip.split()
                    if len(parts) >= 3:
                        properties.append(parts[-1])
                elif line_strip == 'end_header':
                    break

            if vertex_count is None:
                body = f.readlines()
                data_lines = [l.strip() for l in body if l.strip() != '']
                vertex_count = len(data_lines)
                data_iter = iter(data_lines)
            else:
                data_lines = []
                for _ in range(vertex_count):
                    data_lines.append(f.readline().strip())
                data_iter = iter(data_lines)

        # parse data lines into numpy array
        pts = []
        labels = None
        for ln in data_iter:
            if ln == '':
                continue
            parts = ln.split()
            if len(parts) < 3:
                continue
            try:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            except ValueError:
                continue
            pts.append([x, y, z])

            if labels is None:
                if len(parts) > 3:
                    last = parts[-1]
                    try:
                        lab = int(float(last))
                        labels = [lab]
                    except ValueError:
                        labels = None
                else:
                    labels = None
            else:
                try:
                    labels.append(int(float(parts[-1])))
                except ValueError:
                    labels.append(-1)

        points = np.array(pts, dtype=np.float32)
        if labels is not None:
            labels = np.array(labels, dtype=np.int64)
            if labels.shape[0] != points.shape[0]:
                labels = None

        return {'points': points, 'labels': labels, 'file_path': file_path}


def test(FLAGS):
    """Enhanced testing with ablation studies and comprehensive evaluation."""
    if 'config' not in globals():
        raise RuntimeError("Global 'config' object is required by test(). Please ensure config is defined.")
    cfg = globals()['config']

    test_dataset = Semantic3DDataset('test', cfg)

    device = torch.device(f"cuda:{FLAGS.gpu}" if torch.cuda.is_available() else "cpu")

    try:
        model = Network(cfg)
    except NameError:
        raise RuntimeError("Network class is not defined in the scope. Cannot instantiate model for testing.")
    model.to(device)

    if FLAGS.model_path and os.path.exists(FLAGS.model_path):
        checkpoint = torch.load(FLAGS.model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and any(k in checkpoint for k in ['state_dict', 'model']):
            state = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
        else:
            state = checkpoint
        try:
            model.load_state_dict(state)
        except Exception as e:
            new_state = {}
            for k, v in state.items():
                nk = k.replace('module.', '') if k.startswith('module.') else k
                new_state[nk] = v
            model.load_state_dict(new_state)
        print(f"Model restored from {FLAGS.model_path}")
        if isinstance(checkpoint, dict):
            print(f"Loaded model metrics: {checkpoint.get('metrics', {})}")
    else:
        print("Warning: No model checkpoint provided or path does not exist.")
        return

    if FLAGS.ablation:
        print("\nRunning ablation study...")
        test_loader = None
        try:
            res = get_dataloader('test', cfg)
            if isinstance(res, tuple):
                test_loader = res[0]
            else:
                test_loader = res
        except TypeError:
            try:
                res = get_dataloader('test')
                if isinstance(res, tuple):
                    test_loader = res[0]
                else:
                    test_loader = res
            except Exception:
                test_loader = None

        if test_loader is None:
            from torch.utils.data import DataLoader
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        try:
            ablation = AblationStudy(model, cfg)
            ablation_results = ablation.run_ablation(test_loader, device)
            analysis = ablation.analyze_results(ablation_results)
        except NameError:
            print("AblationStudy is not available in the current scope. Skipping ablation.")
            return

        print("\nAblation Study Results:")
        print("=" * 50)
        for strategy, metrics in ablation_results.items():
            print(f"\n{strategy.upper()} Sampling:")
            print(f"  Overall mIoU: {metrics.get('mIoU', float('nan')):.4f}")
            print(f"  Boundary mIoU: {metrics.get('boundary_mIoU', float('nan')):.4f}")
            print(f"  Boundary mF1: {metrics.get('boundary_mF1', float('nan')):.4f}")

        print("\nImprovement Analysis:")
        print("=" * 50)
        for key, value in analysis.items():
            print(f"{key}: {value:.2f}%")

        try:
            visualizer = BoundaryVisualization()
            visualizer.plot_metrics_comparison(ablation_results, 'ablation_study')
        except NameError:
            print("BoundaryVisualization not found; skipping plotting of ablation results.")
    else:
        try:
            tester = ModelTester(model, cfg)
            try:
                tester.test(test_dataset, device=device)
            except TypeError:
                tester.test(model, test_dataset)
        except NameError:
            print("ModelTester class not found; running simple inference loop over test dataset.")
            from torch.utils.data import DataLoader
            loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
            model.eval()
            predictions = []
            with torch.no_grad():
                for batch in loader:
                    pts = batch.get('points')
                    if pts is None:
                        continue
                    pts_t = torch.from_numpy(pts).float().to(device)
                    if pts_t.dim() == 2:
                        inp = pts_t.unsqueeze(0)
                    else:
                        inp = pts_t
                    try:
                        out = model(inp)
                    except Exception as e:
                        print(f"Inference failed for a batch: {e}")
                        continue
                    if isinstance(out, torch.Tensor):
                        preds = out.cpu().numpy()
                    elif isinstance(out, (list, tuple)) and isinstance(out[0], torch.Tensor):
                        preds = out[0].cpu().numpy()
                    else:
                        preds = out
                    predictions.append(preds)
            print(f"Inference completed on {len(test_dataset)} samples (predictions collected).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', type=str, default='train', help='train or test')
    parser.add_argument('--model_path', '-p', type=str, default=None, help='path to the trained model')
    parser.add_argument('--ablation', '-a', action='store_true', help='run ablation study during testing')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID to use')

    FLAGS = parser.parse_args()

    if FLAGS.mode == 'train':
        train(FLAGS)
    elif FLAGS.mode in ['test', 'eval', 'evaluate']:
        test(FLAGS)
    else:
        raise ValueError(f"Unknown mode: {FLAGS.mode}. Supported modes: train, test")

if __name__ == '__main__':
    main()






# from helper_tool import ConfigSemantic3D as cfg
# from helper_tool import DataProcessing as DP
# from RandLANet import Network
# from dataset import Semantic3D
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import argparse
# import os
# import numpy as np
# import logging
# from os.path import join
# from tester_Semantic3D import ModelTester
# from boundary_metrics import BoundaryAwareMetrics, CurvatureAwareEvaluator
# from ablation_study import AblationStudy
# from visualization_tools import BoundaryVisualization
# import wandb  # For experiment tracking (optional but recommended for A* publications)
# import itertools

# def get_dataloader(mode):
#     dataset = Semantic3D(mode)
    
#     # Ensure a valid dataset was created
#     if len(dataset) == 0:
#         logging.warning(f"DataLoader for mode '{mode}' has 0 samples. Check your dataset path and splits.")
#         # Return an empty loader to prevent crashes
#         return DataLoader(dataset, batch_size=cfg.batch_size), dataset

#     dataloader = DataLoader(
#         dataset,
#         batch_size=cfg.batch_size if mode == 'training' else cfg.val_batch_size,
#         shuffle=(mode == 'training'), # Only shuffle training data
#         num_workers=4,
#         collate_fn=dataset.collate_fn # Use the static collate function
#     )
#     return dataloader, dataset

# def train(FLAGS):
#     # Initialize experiment tracking (optional but recommended)
#     # wandb.init(project="gas-3d-segmentation", config=cfg.__dict__)

#     train_loader, _ = get_dataloader('training')
#     val_loader, _ = get_dataloader('validation')

#     device = torch.device(f"cuda:{FLAGS.gpu}" if torch.cuda.is_available() else "cpu")
#     model = Network(cfg)
#     model.to(device)

#     print("DEBUG A | device:", device)
#     print("DEBUG A | param device:", next(model.parameters()).device)
#     print("DEBUG A | model.training (should be True):", model.training)
#     print("DEBUG A | any param.requires_grad:", any(p.requires_grad for p in model.parameters()))

#     optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=cfg.lr_decays.get(0, 0.95))

#     # Get class weights for loss function
#     class_weights = torch.from_numpy(DP.get_class_weights('Semantic3D').squeeze()).float().to(device)

#     # --- FIX: Set a safe ignore index for the loss function ---
#     # Use a value outside the range of possible class indices [0, num_classes-1]
#     ignored_label_index = 255
#     loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignored_label_index)

#     # Initialize metrics
#     boundary_metrics = BoundaryAwareMetrics(cfg.num_classes)
#     curvature_evaluator = CurvatureAwareEvaluator(cfg.num_classes)
#     visualizer = BoundaryVisualization()

#     best_boundary_miou = 0.0

#     for epoch in range(cfg.max_epoch):
#         model.train()
#         train_loss = 0.0

#         for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.max_epoch}")):
#             # Move data to device
#             for key in batch:
#                 if isinstance(batch[key], torch.Tensor):
#                     batch[key] = batch[key].to(device)
            
#             print(f"DEBUG B | iter {i} | batch keys: {list(batch.keys())}")
#             try:
#                 print("DEBUG B | shapes xyz, features, labels:", batch['xyz'].shape, batch['features'].shape, batch['labels'].shape)
#             except Exception as e:
#                 print("DEBUG B | shape error:", e)
#             print("DEBUG B | sample label uniques (orig):", torch.unique(batch['labels']))

#             optimizer.zero_grad()

#             # Forward pass
#             end_points = model(batch)

#             print("DEBUG C1 | logits present keys:", 'logits' in end_points)
#             if 'logits' in end_points:
#                 print("DEBUG C1 | logits.shape:", end_points['logits'].shape)

#             logits = end_points['logits']  # [B, N, C]
#             labels = batch['labels']      # [B, N], contains original labels [0-8]

#             # --- FIX: Remap labels to be compatible with CrossEntropyLoss ---
#             # CrossEntropyLoss expects target labels in the range [0, num_classes-1].
#             # Original labels are 1-8 for classes, and 0 for 'unlabeled'.
#             # We map class labels 1-8 to training labels 0-7.
#             # We map the original 'unlabeled' 0 to our new ignore_index (255).
#             remapped_labels = labels - 1  # Maps 1-8 to 0-7, and 0 to -1
#             remapped_labels[labels == 0] = ignored_label_index # Map original 0 to 255

#             # Flatten logits and remapped labels for loss calculation
#             logits = logits.view(-1, cfg.num_classes)
#             labels = remapped_labels.view(-1)

#             print("DEBUG C2 | loss input shapes:", logits.shape, labels.shape)
#             loss = loss_fn(logits, labels)
#             print("DEBUG C2 | loss:", loss.item())

#             p0 = next(model.parameters()).detach().cpu().clone()

#             loss.backward()

#             # ----------------- DEBUG D (after backward, before step) -----------------
#             n_grads = sum(1 for p in model.parameters() if p.grad is not None)
#             max_grad = max((p.grad.abs().max().item() for p in model.parameters() if p.grad is not None), default=0.)
#             print("DEBUG D | n params w/ grad:", n_grads, "max grad:", max_grad)

#             optimizer.step()

#             p1 = next(model.parameters()).detach().cpu()
#             print("DEBUG E | max param change:", (p1 - p0).abs().max().item())

#             train_loss += loss.item()

#         scheduler.step()

#         # Validation with boundary-aware metrics
#         if (epoch + 1) % 5 == 0:  # Validate every 5 epochs
#             model.eval()
#             boundary_metrics.reset()
#             curvature_evaluator.reset()

#             with torch.no_grad():
#                 for batch in tqdm(val_loader, desc="Validation"):
#                     # Move data to device
#                     for key in batch:
#                         if isinstance(batch[key], torch.Tensor):
#                            batch[key] = batch[key].to(device)

#                     # Forward pass
#                     end_points = model(batch)
#                     predictions = end_points['logits']

#                     # Update metrics
#                     # boundary_metrics.update(predictions, batch['labels'], batch['xyz'][0])
#                     # curvature_metrics = curvature_evaluator.evaluate(predictions, batch['labels'], batch['xyz'][0])
#                     boundary_metrics.update(predictions, batch['labels'], batch['xyz'])
#                     curvature_metrics = curvature_evaluator.evaluate(predictions, batch['labels'], batch['xyz'])

#             # Compute metrics
#             metrics = boundary_metrics.compute_metrics()

#             print(f"\nEpoch {epoch+1} Validation Results:")
#             print(f"Overall mIoU: {metrics['mIoU']:.4f}")
#             print(f"Boundary mIoU: {metrics['boundary_mIoU']:.4f}")
#             print(f"Boundary mF1: {metrics['boundary_mF1']:.4f}")
#             print(f"Boundary Improvement Ratio: {metrics.get('boundary_improvement_ratio', 0):.4f}")

#             # Save best model based on boundary mIoU
#             if metrics['boundary_mIoU'] > best_boundary_miou:
#                 best_boundary_miou = metrics['boundary_mIoU']
#                 torch.save({
#                     'epoch': epoch,
#                     'model_state_dict': model.state_dict(),
#                     'optimizer_state_dict': optimizer.state_dict(),
#                     'metrics': metrics
#                 }, f'best_model_boundary_miou_{best_boundary_miou:.4f}.pth')
#                 print(f"Saved best model with boundary mIoU: {best_boundary_miou:.4f}")

#             # Log to wandb if using
#             # wandb.log(metrics)

#         # Periodic visualization
#         if (epoch + 1) % 10 == 0:
#             model.eval()
#             with torch.no_grad():
#                 # Get one batch for visualization
#                 batch = next(iter(val_loader))
#                 for key in batch:
#                     if isinstance(batch[key], torch.Tensor):
#                         batch[key] = batch[key].to(device)

#                 # Enable auxiliary info storage
#                 if hasattr(model, 'store_aux_info'):
#                     model.store_aux_info = True
#                     model.aux_info = []

#                 end_points = model(batch)
#                 predictions = end_points['logits'].argmax(dim=-1)

#                 # Visualize first sample in batch
#                 visualizer.visualize_boundary_errors(
#                     batch['xyz'][0].cpu(),
#                     predictions[0].cpu(),
#                     batch['labels'][0].cpu(),
#                     f'epoch_{epoch+1}',
#                     cfg.num_classes
#                 )

#                 # Visualize sampling distribution for first layer
#                 if hasattr(model, 'aux_info') and model.aux_info:
#                     visualizer.visualize_sampling_distribution(
#                         batch['xyz'][0].cpu(),
#                         model.aux_info[0]['learned_indices'][0].cpu(),
#                         {k: v[0].cpu() for k, v in model.aux_info[0].items()},
#                         f'epoch_{epoch+1}',
#                         0
#                     )
                
#                 if hasattr(model, 'store_aux_info'):
#                     model.store_aux_info = False

# # #####################################################################################
# # ######################### PyTorch Dataset Implementation ############################
# # #####################################################################################

# class Semantic3DDataset(torch.utils.data.Dataset):
#     def __init__(self, mode, config):
#         self.mode = mode
#         self.config = config
#         self.sub_pc_folder = join(config.data_path, 'input_{:.3f}'.format(config.sub_grid_size))

#         # Use the train/val split from the original RandLA-Net paper for consistency
#         self.all_files = np.sort([join(self.sub_pc_folder, f) for f in os.listdir(self.sub_pc_folder) if f.endswith('.ply')])
#         # This mapping is specific to the reduced benchmark of Semantic3D
#         self.all_splits = [0, 1, 4, 5, 3, 4, 3, 0, 1, 2, 3, 4, 2, 0, 5]
#         self.val_split_id = 1

#         self.files = []
#         if self.mode == 'training':
#             self.files = [f for i, f in enumerate(self.all_files) if self.all_splits[i] != self.val_split_id]
#         elif self.mode in ['validation', 'test']:
#             self.files = [f for i, f in enumerate(self.all_files) if self.all_splits[i] == self.val_split_id]
#         else:
#             raise ValueError(f"Invalid mode: {self.mode}")

#         logging.info(f"Found {len(self.files)} files for {mode}ing.")

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, index):
#         """Load a .ply file (ASCII) and return a dict with points and optional labels.

#         Returns:
#             dict with keys:
#                 'points' : numpy array of shape (N,3)
#                 'labels' : numpy array of shape (N,) or None
#                 'file_path' : original file path
#         """
#         file_path = self.files[index]

#         # Basic ASCII PLY reader robust for common Semantic3D files
#         with open(file_path, 'r') as f:
#             header_lines = []
#             line = f.readline()
#             if not line.startswith('ply'):
#                 raise ValueError(f"Unsupported or corrupted PLY file: {file_path}")
#             header_lines.append(line)
#             vertex_count = None
#             properties = []
#             while True:
#                 line = f.readline()
#                 if not line:
#                     raise ValueError(f"Unexpected end of file while reading header: {file_path}")
#                 header_lines.append(line)
#                 line_strip = line.strip()
#                 if line_strip.startswith('element vertex'):
#                     # example: element vertex 12345
#                     parts = line_strip.split()
#                     if len(parts) >= 3:
#                         try:
#                             vertex_count = int(parts[2])
#                         except ValueError:
#                             vertex_count = None
#                 elif line_strip.startswith('property'):
#                     # collect property names; example: property float x
#                     parts = line_strip.split()
#                     if len(parts) >= 3:
#                         properties.append(parts[-1])
#                 elif line_strip == 'end_header':
#                     break

#             if vertex_count is None:
#                 # fallback: read the rest and count lines (not ideal but better than failing)
#                 body = f.readlines()
#                 data_lines = [l.strip() for l in body if l.strip() != '']
#                 vertex_count = len(data_lines)
#                 data_iter = iter(data_lines)
#             else:
#                 # read exactly vertex_count lines
#                 data_lines = []
#                 for _ in range(vertex_count):
#                     data_lines.append(f.readline().strip())
#                 data_iter = iter(data_lines)

#         # parse data lines into numpy array
#         pts = []
#         labels = None
#         for ln in data_iter:
#             if ln == '':
#                 continue
#             parts = ln.split()
#             # Ensure we have at least 3 numeric coordinates
#             if len(parts) < 3:
#                 continue
#             try:
#                 x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
#             except ValueError:
#                 # skip malformed line
#                 continue
#             pts.append([x, y, z])

#             # Attempt to extract a label if present in properties or extra columns
#             # Heuristic: if properties include 'label' or 'class' or if there are >= 4 columns and the last is integer.
#             if labels is None:
#                 if len(parts) > 3:
#                     last = parts[-1]
#                     try:
#                         lab = int(float(last))
#                         labels = [lab]
#                     except ValueError:
#                         labels = None
#                 else:
#                     labels = None
#             else:
#                 try:
#                     labels.append(int(float(parts[-1])))
#                 except ValueError:
#                     labels.append(-1)

#         points = np.array(pts, dtype=np.float32)
#         if labels is not None:
#             labels = np.array(labels, dtype=np.int64)
#             if labels.shape[0] != points.shape[0]:
#                 # if mismatch, drop labels
#                 labels = None

#         return {'points': points, 'labels': labels, 'file_path': file_path}


# def test(FLAGS):
#     """Enhanced testing with ablation studies and comprehensive evaluation."""
#     # Expect a global `config` variable in the script; if not present, raise informative error.
#     if 'config' not in globals():
#         raise RuntimeError("Global 'config' object is required by test(). Please ensure config is defined.")
#     cfg = globals()['config']

#     test_dataset = Semantic3DDataset('test', cfg)

#     device = torch.device(f"cuda:{FLAGS.gpu}" if torch.cuda.is_available() else "cpu")

#     # Instantiate model (expecting a Network class defined in the script)
#     try:
#         model = Network(cfg)
#     except NameError:
#         raise RuntimeError("Network class is not defined in the scope. Cannot instantiate model for testing.")
#     model.to(device)

#     # Load the trained model checkpoint
#     if FLAGS.model_path and os.path.exists(FLAGS.model_path):
#         checkpoint = torch.load(FLAGS.model_path, map_location=device)
#         # Support both single-state_dict and wrapped checkpoint formats
#         if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
#             state = checkpoint['model_state_dict']
#         elif isinstance(checkpoint, dict) and any(k in checkpoint for k in ['state_dict', 'model']):
#             # try common alternative keys
#             state = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
#         else:
#             state = checkpoint
#         try:
#             model.load_state_dict(state)
#         except Exception as e:
#             # attempt to handle DataParallel prefixed keys
#             new_state = {}
#             for k, v in state.items():
#                 nk = k.replace('module.', '') if k.startswith('module.') else k
#                 new_state[nk] = v
#             model.load_state_dict(new_state)
#         print(f"Model restored from {FLAGS.model_path}")
#         if isinstance(checkpoint, dict):
#             print(f"Loaded model metrics: {checkpoint.get('metrics', {})}")
#     else:
#         print("Warning: No model checkpoint provided or path does not exist.")
#         return

#     # Run comprehensive evaluation
#     if FLAGS.ablation:
#         print("\nRunning ablation study...")
#         # Try to obtain a dataloader; support multiple get_dataloader signatures
#         test_loader = None
#         try:
#             # try signature: get_dataloader(mode, config)
#             res = get_dataloader('test', cfg)
#             if isinstance(res, tuple):
#                 test_loader = res[0]
#             else:
#                 test_loader = res
#         except TypeError:
#             try:
#                 res = get_dataloader('test')
#                 if isinstance(res, tuple):
#                     test_loader = res[0]
#                 else:
#                     test_loader = res
#             except Exception:
#                 test_loader = None

#         if test_loader is None:
#             # fallback to building a simple DataLoader
#             from torch.utils.data import DataLoader
#             test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

#         try:
#             ablation = AblationStudy(model, cfg)
#             ablation_results = ablation.run_ablation(test_loader, device)
#             analysis = ablation.analyze_results(ablation_results)
#         except NameError:
#             print("AblationStudy is not available in the current scope. Skipping ablation.")
#             return

#         print("\nAblation Study Results:")
#         print("=" * 50)
#         for strategy, metrics in ablation_results.items():
#             print(f"\n{strategy.upper()} Sampling:")
#             print(f"  Overall mIoU: {metrics.get('mIoU', float('nan')):.4f}")
#             print(f"  Boundary mIoU: {metrics.get('boundary_mIoU', float('nan')):.4f}")
#             print(f"  Boundary mF1: {metrics.get('boundary_mF1', float('nan')):.4f}")

#         print("\nImprovement Analysis:")
#         print("=" * 50)
#         for key, value in analysis.items():
#             print(f"{key}: {value:.2f}%")

#         # Create comparison plots if visualizer available
#         try:
#             visualizer = BoundaryVisualization()
#             visualizer.plot_metrics_comparison(ablation_results, 'ablation_study')
#         except NameError:
#             print("BoundaryVisualization not found; skipping plotting of ablation results.")
#     else:
#         # Standard testing using provided ModelTester if present
#         try:
#             tester = ModelTester(model, cfg)
#             # Prefer tester.test(dataset, device) if available
#             try:
#                 tester.test(test_dataset, device=device)
#             except TypeError:
#                 # fallback to tester.test(model, dataset)
#                 tester.test(model, test_dataset)
#         except NameError:
#             # Fallback: simple inference loop
#             print("ModelTester class not found; running simple inference loop over test dataset.")
#             from torch.utils.data import DataLoader
#             loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
#             model.eval()
#             predictions = []
#             with torch.no_grad():
#                 for batch in loader:
#                     pts = batch.get('points')
#                     if pts is None:
#                         continue
#                     pts_t = torch.from_numpy(pts).float().to(device)
#                     # If model expects additional dims/batch, we attempt to add batch dim
#                     if pts_t.dim() == 2:
#                         inp = pts_t.unsqueeze(0)
#                     else:
#                         inp = pts_t
#                     try:
#                         out = model(inp)
#                     except Exception as e:
#                         print(f"Inference failed for a batch: {e}")
#                         continue
#                     # Attempt to extract prediction tensor/array
#                     if isinstance(out, torch.Tensor):
#                         preds = out.cpu().numpy()
#                     elif isinstance(out, (list, tuple)) and isinstance(out[0], torch.Tensor):
#                         preds = out[0].cpu().numpy()
#                     else:
#                         preds = out
#                     predictions.append(preds)
#             print(f"Inference completed on {len(test_dataset)} samples (predictions collected).")


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--mode', '-m', type=str, default='train', help='train or test')
#     parser.add_argument('--model_path', '-p', type=str, default=None, help='path to the trained model')
#     parser.add_argument('--ablation', '-a', action='store_true', help='run ablation study during testing')
#     parser.add_argument('--gpu', type=int, default=0, help='GPU device ID to use')

#     FLAGS = parser.parse_args()

#     if FLAGS.mode == 'train':
#         train(FLAGS)
#     elif FLAGS.mode in ['test', 'eval', 'evaluate']:
#         test(FLAGS)
#     else:
#         raise ValueError(f"Unknown mode: {FLAGS.mode}. Supported modes: train, test")

# if __name__ == '__main__':
#     main()















# from helper_tool import ConfigSemantic3D as cfg
# from helper_tool import DataProcessing as DP
# from RandLANet import Network
# from dataset import Semantic3D
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import argparse
# import os
# import numpy as np
# import logging
# from os.path import join
# from tester_Semantic3D import ModelTester
# from boundary_metrics import BoundaryAwareMetrics, CurvatureAwareEvaluator
# from ablation_study import AblationStudy
# from visualization_tools import BoundaryVisualization
# import wandb  # For experiment tracking (optional but recommended for A* publications)

# def get_dataloader(mode):
#     dataset = Semantic3D(mode)
    
#     dataloader = DataLoader(
#         dataset,
#         batch_size=cfg.batch_size if mode == 'training' else cfg.val_batch_size,
#         # CORRECTED: Only shuffle the training data
#         shuffle=(mode == 'training'),
#         num_workers=4,
#         # CORRECTED: Use the instance method for collate_fn, which is safer
#         collate_fn=dataset.collate_fn
#     )
#     return dataloader, dataset

# def train(FLAGS):
#     # Initialize experiment tracking (optional but recommended)
#     # wandb.init(project="gas-3d-segmentation", config=cfg.__dict__)

#     train_loader, _ = get_dataloader('training')
#     val_loader, _ = get_dataloader('validation')

#     device = torch.device(f"cuda:{FLAGS.gpu}" if torch.cuda.is_available() else "cpu")
#     model = Network(cfg)
#     model.to(device)

#     optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=cfg.lr_decays.get(0, 0.95))

#     # Get class weights for loss function
#     class_weights = torch.from_numpy(DP.get_class_weights('Semantic3D').squeeze()).float().to(device)

#     # Use the first ignored label index from config
#     # In Semantic3D, the ignored label is 0, which corresponds to index 0
#     ignored_label_index = cfg.ignored_label_inds[0]
#     loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignored_label_index)

#     # Initialize metrics
#     boundary_metrics = BoundaryAwareMetrics(cfg.num_classes)
#     curvature_evaluator = CurvatureAwareEvaluator(cfg.num_classes)
#     visualizer = BoundaryVisualization()

#     best_boundary_miou = 0.0

#     for epoch in range(cfg.max_epoch):
#         model.train()
#         train_loss = 0.0

#         for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.max_epoch}")):
#             # Move data to device
#             for key in batch:
#                 if isinstance(batch[key], torch.Tensor):
#                     batch[key] = batch[key].to(device)

#             optimizer.zero_grad()

#             # Forward pass
#             end_points = model(batch) # <<< FIX: Added missing forward pass
#             logits = end_points['logits']  # [B, N, C]
#             labels = batch['labels']      # [B, N]

#             # Flatten logits and labels
#             logits = logits.view(-1, cfg.num_classes) # <<< FIX: Changed config to cfg
#             labels = labels.view(-1)

#             loss = loss_fn(logits, labels)
#             loss.backward()
#             optimizer.step()

#             train_loss += loss.item()

#         scheduler.step()

#         # Validation with boundary-aware metrics
#         if (epoch + 1) % 5 == 0:  # Validate every 5 epochs
#             model.eval()
#             boundary_metrics.reset()
#             curvature_evaluator.reset()

#             with torch.no_grad():
#                 for batch in tqdm(val_loader, desc="Validation"):
#                     # Move data to device
#                     for key in batch:
#                         if isinstance(batch[key], torch.Tensor):
#                            batch[key] = batch[key].to(device)

#                     # Forward pass
#                     end_points = model(batch)
#                     predictions = end_points['logits']

#                     # Update metrics
#                     boundary_metrics.update(predictions, batch['labels'], batch['xyz'][0])
#                     curvature_metrics = curvature_evaluator.evaluate(predictions, batch['labels'], batch['xyz'][0])

#             # Compute metrics
#             metrics = boundary_metrics.compute_metrics()

#             print(f"\nEpoch {epoch+1} Validation Results:")
#             print(f"Overall mIoU: {metrics['mIoU']:.4f}")
#             print(f"Boundary mIoU: {metrics['boundary_mIoU']:.4f}")
#             print(f"Boundary mF1: {metrics['boundary_mF1']:.4f}")
#             print(f"Boundary Improvement Ratio: {metrics.get('boundary_improvement_ratio', 0):.4f}")

#             # Save best model based on boundary mIoU
#             if metrics['boundary_mIoU'] > best_boundary_miou:
#                 best_boundary_miou = metrics['boundary_mIoU']
#                 torch.save({
#                     'epoch': epoch,
#                     'model_state_dict': model.state_dict(),
#                     'optimizer_state_dict': optimizer.state_dict(),
#                     'metrics': metrics
#                 }, f'best_model_boundary_miou_{best_boundary_miou:.4f}.pth')
#                 print(f"Saved best model with boundary mIoU: {best_boundary_miou:.4f}")

#             # Log to wandb if using
#             # wandb.log(metrics)

#         # Periodic visualization
#         if (epoch + 1) % 10 == 0:
#             model.eval()
#             with torch.no_grad():
#                 # Get one batch for visualization
#                 batch = next(iter(val_loader))
#                 for key in batch:
#                     if isinstance(batch[key], torch.Tensor):
#                         batch[key] = batch[key].to(device)

#                 # Enable auxiliary info storage
#                 if hasattr(model, 'store_aux_info'):
#                     model.store_aux_info = True
#                     model.aux_info = []

#                 end_points = model(batch)
#                 predictions = end_points['logits'].argmax(dim=-1)

#                 # Visualize first sample in batch
#                 visualizer.visualize_boundary_errors(
#                     batch['xyz'][0].cpu(),
#                     predictions[0].cpu(),
#                     batch['labels'][0].cpu(),
#                     f'epoch_{epoch+1}',
#                     cfg.num_classes
#                 )

#                 # Visualize sampling distribution for first layer
#                 if hasattr(model, 'aux_info') and model.aux_info:
#                     visualizer.visualize_sampling_distribution(
#                         batch['xyz'][0].cpu(),
#                         model.aux_info[0]['learned_indices'][0].cpu(),
#                         {k: v[0].cpu() for k, v in model.aux_info[0].items()},
#                         f'epoch_{epoch+1}',
#                         0
#                     )
                
#                 if hasattr(model, 'store_aux_info'):
#                     model.store_aux_info = False

# # #####################################################################################
# # ######################### PyTorch Dataset Implementation ############################
# # #####################################################################################

# class Semantic3DDataset(torch.utils.data.Dataset):
#     def __init__(self, mode, config):
#         self.mode = mode
#         self.config = config
#         self.sub_pc_folder = join(config.data_path, 'input_{:.3f}'.format(config.sub_grid_size))

#         # Use the train/val split from the original RandLA-Net paper for consistency
#         self.all_files = np.sort([join(self.sub_pc_folder, f) for f in os.listdir(self.sub_pc_folder) if f.endswith('.ply')])
#         # This mapping is specific to the reduced benchmark of Semantic3D
#         self.all_splits = [0, 1, 4, 5, 3, 4, 3, 0, 1, 2, 3, 4, 2, 0, 5]
#         self.val_split_id = 1

#         self.files = []
#         if self.mode == 'training':
#             self.files = [f for i, f in enumerate(self.all_files) if self.all_splits[i] != self.val_split_id]
#         elif self.mode in ['validation', 'test']:
#             self.files = [f for i, f in enumerate(self.all_files) if self.all_splits[i] == self.val_split_id]
#         else:
#             raise ValueError(f"Invalid mode: {self.mode}")

#         logging.info(f"Found {len(self.files)} files for {mode}ing.")

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, index):
#         """Load a .ply file (ASCII) and return a dict with points and optional labels.

#         Returns:
#             dict with keys:
#                 'points' : numpy array of shape (N,3)
#                 'labels' : numpy array of shape (N,) or None
#                 'file_path' : original file path
#         """
#         file_path = self.files[index]

#         # Basic ASCII PLY reader robust for common Semantic3D files
#         with open(file_path, 'r') as f:
#             header_lines = []
#             line = f.readline()
#             if not line.startswith('ply'):
#                 raise ValueError(f"Unsupported or corrupted PLY file: {file_path}")
#             header_lines.append(line)
#             vertex_count = None
#             properties = []
#             while True:
#                 line = f.readline()
#                 if not line:
#                     raise ValueError(f"Unexpected end of file while reading header: {file_path}")
#                 header_lines.append(line)
#                 line_strip = line.strip()
#                 if line_strip.startswith('element vertex'):
#                     # example: element vertex 12345
#                     parts = line_strip.split()
#                     if len(parts) >= 3:
#                         try:
#                             vertex_count = int(parts[2])
#                         except ValueError:
#                             vertex_count = None
#                 elif line_strip.startswith('property'):
#                     # collect property names; example: property float x
#                     parts = line_strip.split()
#                     if len(parts) >= 3:
#                         properties.append(parts[-1])
#                 elif line_strip == 'end_header':
#                     break

#             if vertex_count is None:
#                 # fallback: read the rest and count lines (not ideal but better than failing)
#                 body = f.readlines()
#                 data_lines = [l.strip() for l in body if l.strip() != '']
#                 vertex_count = len(data_lines)
#                 data_iter = iter(data_lines)
#             else:
#                 # read exactly vertex_count lines
#                 data_lines = []
#                 for _ in range(vertex_count):
#                     data_lines.append(f.readline().strip())
#                 data_iter = iter(data_lines)

#         # parse data lines into numpy array
#         pts = []
#         labels = None
#         for ln in data_iter:
#             if ln == '':
#                 continue
#             parts = ln.split()
#             # Ensure we have at least 3 numeric coordinates
#             if len(parts) < 3:
#                 continue
#             try:
#                 x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
#             except ValueError:
#                 # skip malformed line
#                 continue
#             pts.append([x, y, z])

#             # Attempt to extract a label if present in properties or extra columns
#             # Heuristic: if properties include 'label' or 'class' or if there are >= 4 columns and the last is integer.
#             if labels is None:
#                 if len(parts) > 3:
#                     last = parts[-1]
#                     try:
#                         lab = int(float(last))
#                         labels = [lab]
#                     except ValueError:
#                         labels = None
#                 else:
#                     labels = None
#             else:
#                 try:
#                     labels.append(int(float(parts[-1])))
#                 except ValueError:
#                     labels.append(-1)

#         points = np.array(pts, dtype=np.float32)
#         if labels is not None:
#             labels = np.array(labels, dtype=np.int64)
#             if labels.shape[0] != points.shape[0]:
#                 # if mismatch, drop labels
#                 labels = None

#         return {'points': points, 'labels': labels, 'file_path': file_path}


# def test(FLAGS):
#     """Enhanced testing with ablation studies and comprehensive evaluation."""
#     # Expect a global `config` variable in the script; if not present, raise informative error.
#     if 'config' not in globals():
#         raise RuntimeError("Global 'config' object is required by test(). Please ensure config is defined.")
#     cfg = globals()['config']

#     test_dataset = Semantic3DDataset('test', cfg)

#     device = torch.device(f"cuda:{FLAGS.gpu}" if torch.cuda.is_available() else "cpu")

#     # Instantiate model (expecting a Network class defined in the script)
#     try:
#         model = Network(cfg)
#     except NameError:
#         raise RuntimeError("Network class is not defined in the scope. Cannot instantiate model for testing.")
#     model.to(device)

#     # Load the trained model checkpoint
#     if FLAGS.model_path and os.path.exists(FLAGS.model_path):
#         checkpoint = torch.load(FLAGS.model_path, map_location=device)
#         # Support both single-state_dict and wrapped checkpoint formats
#         if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
#             state = checkpoint['model_state_dict']
#         elif isinstance(checkpoint, dict) and any(k in checkpoint for k in ['state_dict', 'model']):
#             # try common alternative keys
#             state = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
#         else:
#             state = checkpoint
#         try:
#             model.load_state_dict(state)
#         except Exception as e:
#             # attempt to handle DataParallel prefixed keys
#             new_state = {}
#             for k, v in state.items():
#                 nk = k.replace('module.', '') if k.startswith('module.') else k
#                 new_state[nk] = v
#             model.load_state_dict(new_state)
#         print(f"Model restored from {FLAGS.model_path}")
#         if isinstance(checkpoint, dict):
#             print(f"Loaded model metrics: {checkpoint.get('metrics', {})}")
#     else:
#         print("Warning: No model checkpoint provided or path does not exist.")
#         return

#     # Run comprehensive evaluation
#     if FLAGS.ablation:
#         print("\nRunning ablation study...")
#         # Try to obtain a dataloader; support multiple get_dataloader signatures
#         test_loader = None
#         try:
#             # try signature: get_dataloader(mode, config)
#             res = get_dataloader('test', cfg)
#             if isinstance(res, tuple):
#                 test_loader = res[0]
#             else:
#                 test_loader = res
#         except TypeError:
#             try:
#                 res = get_dataloader('test')
#                 if isinstance(res, tuple):
#                     test_loader = res[0]
#                 else:
#                     test_loader = res
#             except Exception:
#                 test_loader = None

#         if test_loader is None:
#             # fallback to building a simple DataLoader
#             from torch.utils.data import DataLoader
#             test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

#         try:
#             ablation = AblationStudy(model, cfg)
#             ablation_results = ablation.run_ablation(test_loader, device)
#             analysis = ablation.analyze_results(ablation_results)
#         except NameError:
#             print("AblationStudy is not available in the current scope. Skipping ablation.")
#             return

#         print("\nAblation Study Results:")
#         print("=" * 50)
#         for strategy, metrics in ablation_results.items():
#             print(f"\n{strategy.upper()} Sampling:")
#             print(f"  Overall mIoU: {metrics.get('mIoU', float('nan')):.4f}")
#             print(f"  Boundary mIoU: {metrics.get('boundary_mIoU', float('nan')):.4f}")
#             print(f"  Boundary mF1: {metrics.get('boundary_mF1', float('nan')):.4f}")

#         print("\nImprovement Analysis:")
#         print("=" * 50)
#         for key, value in analysis.items():
#             print(f"{key}: {value:.2f}%")

#         # Create comparison plots if visualizer available
#         try:
#             visualizer = BoundaryVisualization()
#             visualizer.plot_metrics_comparison(ablation_results, 'ablation_study')
#         except NameError:
#             print("BoundaryVisualization not found; skipping plotting of ablation results.")
#     else:
#         # Standard testing using provided ModelTester if present
#         try:
#             tester = ModelTester(model, cfg)
#             # Prefer tester.test(dataset, device) if available
#             try:
#                 tester.test(test_dataset, device=device)
#             except TypeError:
#                 # fallback to tester.test(model, dataset)
#                 tester.test(model, test_dataset)
#         except NameError:
#             # Fallback: simple inference loop
#             print("ModelTester class not found; running simple inference loop over test dataset.")
#             from torch.utils.data import DataLoader
#             loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
#             model.eval()
#             predictions = []
#             with torch.no_grad():
#                 for batch in loader:
#                     pts = batch.get('points')
#                     if pts is None:
#                         continue
#                     pts_t = torch.from_numpy(pts).float().to(device)
#                     # If model expects additional dims/batch, we attempt to add batch dim
#                     if pts_t.dim() == 2:
#                         inp = pts_t.unsqueeze(0)
#                     else:
#                         inp = pts_t
#                     try:
#                         out = model(inp)
#                     except Exception as e:
#                         print(f"Inference failed for a batch: {e}")
#                         continue
#                     # Attempt to extract prediction tensor/array
#                     if isinstance(out, torch.Tensor):
#                         preds = out.cpu().numpy()
#                     elif isinstance(out, (list, tuple)) and isinstance(out[0], torch.Tensor):
#                         preds = out[0].cpu().numpy()
#                     else:
#                         preds = out
#                     predictions.append(preds)
#             print(f"Inference completed on {len(test_dataset)} samples (predictions collected).")


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--mode', '-m', type=str, default='train', help='train or test')
#     parser.add_argument('--model_path', '-p', type=str, default=None, help='path to the trained model')
#     parser.add_argument('--ablation', '-a', action='store_true', help='run ablation study during testing')
#     parser.add_argument('--gpu', type=int, default=0, help='GPU device ID to use')

#     FLAGS = parser.parse_args()

#     if FLAGS.mode == 'train':
#         train(FLAGS)
#     elif FLAGS.mode in ['test', 'eval', 'evaluate']:
#         test(FLAGS)
#     else:
#         raise ValueError(f"Unknown mode: {FLAGS.mode}. Supported modes: train, test")

# if __name__ == '__main__':
#     main()