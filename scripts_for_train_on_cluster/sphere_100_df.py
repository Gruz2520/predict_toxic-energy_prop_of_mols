from dig.threedgraph.dataset import QM93D
from dig.threedgraph.method import SphereNet, ComENet, DimeNetPP, ProNet, SchNet
from dig.threedgraph.evaluation import ThreeDEvaluator
from dig.threedgraph.method import run
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_sparse import SparseTensor
import utils
import tts

name_of_model = 'shere_100_df'
print(name_of_model, flush=True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")

dataset = utils.clean_dataset(pd.read_csv("pred_toxic/dataset/md_dataset_full_v3.csv"))

X_train, X_val, X_test, y_train, y_val, y_test = tts.create_tts_from_df(dataset)

model = SphereNet(energy_and_force=True, cutoff=5.0, num_layers=4,
                  hidden_channels=128, out_channels=1, int_emb_size=64,
                  basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8, out_emb_channels=256,
                  num_spherical=3, num_radial=6, envelope_exponent=5,
                  num_before_skip=1, num_after_skip=2, num_output_layers=3)
loss_func = torch.nn.L1Loss()
evaluation = ThreeDEvaluator()

run3d = run()
run3d.run(device, X_train, X_val, X_test, model, loss_func, evaluation,
          epochs=100, batch_size=32, vt_batch_size=32, lr=0.0005, lr_decay_factor=0.5, lr_decay_step_size=15,
          save_dir=f"pred_toxic/models/{name_of_model}", log_dir=f"pred_toxic/logs/{name_of_model}")