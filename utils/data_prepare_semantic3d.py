from sklearn.neighbors import KDTree
from pathlib import Path
import numpy as np
import pickle
import sys

# Base directories
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
sys.path.extend([str(BASE_DIR), str(ROOT_DIR)])

from helper_ply import write_ply
from helper_tool import DataProcessing as DP

# Parameters
grid_size = 0.06

# <-- Windows-style dataset path; adjust drive/dirs as needed -->
dataset_path = Path(r"D:/RandLA-Net/data/semantic3d/original_data")

# Derive the other folders
original_pc_folder = dataset_path.parent / "original_ply"
sub_pc_folder = dataset_path.parent / "input_{:.3f}".format(grid_size)

# Create if missing
original_pc_folder.mkdir(parents=True, exist_ok=True)
sub_pc_folder.mkdir(parents=True, exist_ok=True)

# Process each .txt file
for pc_path in dataset_path.glob("*.txt"):
    print(pc_path)
    file_stem = pc_path.stem  # filename without extension

    # Skip if KDTree already exists
    tree_file = sub_pc_folder / "{}_KDTree.pkl".format(file_stem)
    if tree_file.exists():
        continue

    # Load point cloud
    pc = DP.load_pc_semantic3d(str(pc_path))

    # Check for labels
    label_path = pc_path.with_suffix(".labels")
    if label_path.exists():
        labels = DP.load_label_semantic3d(str(label_path))
        full_ply = original_pc_folder / "{}.ply".format(file_stem)

        # First subsampling for storage
        sub_pts, sub_cols, sub_lbls = DP.grid_sub_sampling(
            pc[:, :3].astype(np.float32),
            pc[:, 4:7].astype(np.uint8),
            labels, 0.01
        )
        sub_lbls = np.squeeze(sub_lbls)

        write_ply(str(full_ply),
                  (sub_pts, sub_cols, sub_lbls),
                  ["x", "y", "z", "red", "green", "blue", "class"])

        # Second subsampling for KDTree
        xyz2, cols2, lbl2 = DP.grid_sub_sampling(
            sub_pts, sub_cols, sub_lbls, grid_size
        )
        cols2 = cols2 / 255.0
        ply2 = sub_pc_folder / "{}.ply".format(file_stem)
        write_ply(str(ply2),
                  [xyz2, cols2, np.squeeze(lbl2)],
                  ["x", "y", "z", "red", "green", "blue", "class"])

        # Build and save KDTree
        tree = KDTree(xyz2, leaf_size=50)
        with open(str(tree_file), "wb") as f:
            pickle.dump(tree, f)

        # Save projection indices
        proj_idx = tree.query(sub_pts, return_distance=False)
        proj_idx = proj_idx.astype(np.int32).squeeze()
        proj_save = sub_pc_folder / "{}_proj.pkl".format(file_stem)
        with open(str(proj_save), "wb") as f:
            pickle.dump([proj_idx, labels], f)

    else:
        # No labels case
        full_ply = original_pc_folder / "{}.ply".format(file_stem)
        write_ply(str(full_ply),
                  (pc[:, :3].astype(np.float32),
                   pc[:, 4:7].astype(np.uint8)),
                  ["x", "y", "z", "red", "green", "blue"])

        # Subsample + KDTree
        xyz2, cols2 = DP.grid_sub_sampling(
            pc[:, :3].astype(np.float32),
            pc[:, 4:7].astype(np.uint8),
            grid_size=grid_size
        )
        cols2 = cols2 / 255.0

        ply2 = sub_pc_folder / "{}.ply".format(file_stem)
        write_ply(str(ply2),
                  [xyz2, cols2],
                  ["x", "y", "z", "red", "green", "blue"])

        tree = KDTree(xyz2, leaf_size=50)
        with open(str(tree_file), "wb") as f:
            pickle.dump(tree, f)

        proj_idx = tree.query(pc[:, :3].astype(np.float32),
                              return_distance=False)
        proj_idx = proj_idx.astype(np.int32).squeeze()
        proj_save = sub_pc_folder / "{}_proj.pkl".format(file_stem)
        with open(str(proj_save), "wb") as f:
            pickle.dump([proj_idx, np.zeros(len(proj_idx), dtype=np.uint8)], f)
