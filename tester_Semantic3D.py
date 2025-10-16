import torch
import numpy as np
from tqdm import tqdm
from helper_ply import write_ply
from os.path import join, exists
from os import makedirs

class ModelTester:
    def __init__(self, model, dataset, restore_snap=None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.model.to(self.device)
        if restore_snap:
            self.model.load_state_dict(torch.load(restore_snap))
            print(f"Model restored from {restore_snap}")

        self.dataset = dataset
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=4
        )

        self.test_probs = [
            np.zeros((l.data.shape[0], model.config.num_classes), dtype=np.float16)
            for l in dataset.input_trees["test"]
        ]

    def test(self, num_votes=100):
        self.model.eval()
        # Testing logic...
