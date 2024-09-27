from torch_geometric.data import Data
from rdkit import Chem
import torch
from tqdm import tqdm

# Очистка датасета
def clean_dataset(df):
    print(f"Shape of old dataset: {df.shape}")
    print("Processing...")
    df = df[df['mouse_intraperitoneal_LD50'] > 0]
    bads = [18213, 3039, 3091, 7246, 18284, 19060, 27616, 44018, 44311]

    df = df[~df['mol_id'].isin(bads)]
    
    for col in tqdm(df.columns):
        if col.startswith('xyz_'):
            df.loc[:, col] = df[col].str.replace('A', 'Al')
            df.loc[:, col] = df[col].str.replace('B', 'Br')
    print(f"Shape of new dataset: {df.shape}")
    return df

# Самописный конвертер в требуемый формат для модели
def xyz_string_to_graph_data(xyz_string, target):
    lines = xyz_string.strip().split('\n')
    coords = []
    atom_types = []
    for line in lines:
        atom_data = line.strip().split()
        atom_type = atom_data[0]
        x, y, z = map(float, atom_data[1:4])
        coords.append([x, y, z])
        atom_types.append(Chem.Atom(atom_type))
    pos = torch.tensor(coords)
    z = torch.tensor([atom.GetAtomicNum() for atom in atom_types])
    
    return Data(pos=pos, z=z, y=target)

# TODO
# Функция eval для построения красивых метрик
# Разобраться с графиками обучения