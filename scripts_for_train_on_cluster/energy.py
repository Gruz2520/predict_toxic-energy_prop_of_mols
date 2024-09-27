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
from torch_geometric.data import Data, Batch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_sparse import SparseTensor
import random
# название модели, под которым оно будет сохраняться в логах

device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")

def xyz_string_to_graph_data(xyz_string, target, formula):
    lines = xyz_string.strip().split('\n')
    coords = []
    atom_types = []
    for line in lines[2:]:
        atom_data = line.strip().split()
        atom_type = atom_data[0]
        x, y, z = map(float, atom_data[1:4])
        coords.append([x, y, z])
        atom_types.append(Chem.Atom(atom_type))
    pos = torch.tensor(coords)
    z = torch.tensor([atom.GetAtomicNum() for atom in atom_types])
    
    return Data(pos=pos, z=z, y=target, formula=formula)

data = pd.read_csv('pred_toxic/dataset/data_all_new.csv')

molecules = []

for i, row in tqdm(data.iterrows(), total=data.shape[0]):
    mol = Chem.MolFromXYZBlock(row['xyz'])
    formula = Chem.MolToInchi(mol).split('/', )[1]
    
    molecules.append(xyz_string_to_graph_data(row['xyz'],
                    row['U_0'] / mol.GetNumAtoms(),
                    formula))
    
def create_structure_dict(structures: list):
    structure_dict = {}

    for structure in tqdm(structures, total=len(structures)):
        if structure.formula in structure_dict:
            structure_dict[structure.formula].append(structure)
        else:
            structure_dict[structure.formula] = [structure]
        
    return structure_dict

def tts_def(X, val_size=0.2, test_size=0.1):
    train_data, test_val_data = train_test_split(X, test_size=test_size+val_size, random_state=42)
    val_data, test_data = train_test_split(test_val_data, test_size=(test_size/(test_size+val_size)), random_state=42)
    
    return train_data, val_data, test_data

def create_tts_def_en(structure_dict: dict, val_size=0.2, test_size=0.1):
    mol_train, mol_val, mol_test = [], [], []
    targets_train, targets_val, targets_test = [], [], []
    train_all, test_all, val_all = [], [], []

    for _, data in tqdm(structure_dict.items()):
        if len(data) < 4:
            train_all += data
        else:
            train_data, val_data, test_data = tts_def(data, val_size, test_size)
            train_all += train_data
            val_all += val_data
            test_all += test_data
            
    random.shuffle(train_all)
    random.shuffle(val_all)
    random.shuffle(test_all)

    for t_data in train_all:
        mol_train.append(t_data)
        targets_train.append(t_data.y)
    
    for tv_data in val_all:
        mol_val.append(tv_data)
        targets_val.append(tv_data.y)
    
    for te_data in test_all:
        mol_test.append(te_data)
        targets_test.append(te_data.y)
    
    return mol_train, mol_val, mol_test, targets_train, targets_val, targets_test

structure_dict = create_structure_dict(molecules)
X_train, X_val, X_test, y_train, y_val, y_test = create_tts_def_en(structure_dict)

loss_func = torch.nn.L1Loss()
evaluation = ThreeDEvaluator()


name_of_model = 'schenet_300_energy'
print(name_of_model, flush=True)
model = SchNet(energy_and_force=False, cutoff=15.0, num_layers=6, hidden_channels=128, out_channels=1, num_filters=128, num_gaussians=50)
run3d = run()
run3d.run(device, X_train, X_val, X_test, model, loss_func, evaluation,
          epochs=300, batch_size=32, vt_batch_size=32, lr=0.0005, lr_decay_factor=0.5, lr_decay_step_size=15,
          save_dir=f"pred_toxic/models/{name_of_model}", log_dir=f"pred_toxic/logs/{name_of_model}")


name_of_model = 'sphere_300_energy'
print(name_of_model, flush=True)
model = SphereNet(energy_and_force=True, cutoff=15.0, num_layers=4,
                      hidden_channels=128, out_channels=1, int_emb_size=64,
                      basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8, out_emb_channels=256,
                      num_spherical=3, num_radial=6, envelope_exponent=5,
                      num_before_skip=1, num_after_skip=2, num_output_layers=3)


run3d = run()
run3d.run(device, X_train, X_val, X_test, model, loss_func, evaluation,
              epochs=300, batch_size=32, vt_batch_size=32, lr=0.0005, lr_decay_factor=0.5, lr_decay_step_size=15,
              save_dir=f"models/{name_of_model}", log_dir=f"logs/{name_of_model}")


name_of_model = 'dimenetpp_300_energy'
print(name_of_model, flush=True)
model = DimeNetPP(energy_and_force=False, cutoff=15.0, num_layers=4, hidden_channels=128, out_channels=1, int_emb_size=64, basis_emb_size=8, out_emb_channels=256, num_spherical=7, num_radial=6, envelope_exponent=5, num_before_skip=1, num_after_skip=2, num_output_layers=3, output_init='GlorotOrthogonal')

run3d = run()
run3d.run(device, X_train, X_val, X_test, model, loss_func, evaluation,
              epochs=300, batch_size=32, vt_batch_size=32, lr=0.0005, lr_decay_factor=0.5, lr_decay_step_size=15,
              save_dir=f"pred_toxic/models/{name_of_model}", log_dir=f"pred_toxic/logs/{name_of_model}")
    

name_of_model = 'comenet_300_energy'
print(name_of_model, flush=True)
model = ComENet(cutoff=15.0, num_layers=4, hidden_channels=256, middle_channels=64, out_channels=1, num_radial=3, num_spherical=2, num_output_layers=3)

run3d = run()
run3d.run(device, X_train, X_val, X_test, model, loss_func, evaluation,
              epochs=300, batch_size=32, vt_batch_size=32, lr=0.0005, lr_decay_factor=0.5, lr_decay_step_size=15,
              save_dir=f"pred_toxic/models/{name_of_model}", log_dir=f"pred_toxic/logs/{name_of_model}")
