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
import pickle

path = 'pred_toxic/dataset/data'

mols = [1, 5, 10, 15]

for i in mols:
    name_of_model = f'spherenet_100_{i}mol_mv'
    print(name_of_model, flush=True)

    with open(f'{path}/data_train_{i}.pkl', 'rb') as f:
        data_train = pickle.load(f)

    with open(f'{path}/data_validation_{i}.pkl', 'rb') as f:
        data_val = pickle.load(f)

    with open(f'{path}/data_test_{i}.pkl', 'rb') as f:
        data_test = pickle.load(f)

    
    X_train = []
    X_val = []
    X_test = []
    y_train = []
    y_val = []
    y_test = []

    for i in data_train:
        X_train.append(i)
        y_train.append(i.y)

    for i in data_val:
        X_val.append(i)
        y_val.append(i.y)

    for i in data_test:
        X_test.append(i)
        y_test.append(i.y)


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")

    model = SphereNet(energy_and_force=True, cutoff=15.0, num_layers=4,
                      hidden_channels=128, out_channels=1, int_emb_size=64,
                      basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8, out_emb_channels=256,
                      num_spherical=3, num_radial=6, envelope_exponent=5,
                      num_before_skip=1, num_after_skip=2, num_output_layers=3)
    loss_func = torch.nn.L1Loss()
    evaluation = ThreeDEvaluator()

    run3d = run()
    run3d.run(device, X_train, X_val, X_test, model, loss_func, evaluation,
              epochs=100, batch_size=32, vt_batch_size=32, lr=0.0005, lr_decay_factor=0.5, lr_decay_step_size=15,
              save_dir=f"models/{name_of_model}", log_dir=f"logs/{name_of_model}")