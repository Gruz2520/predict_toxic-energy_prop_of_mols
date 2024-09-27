from tqdm import tqdm
import torch
import random
from sklearn.model_selection import train_test_split
from utils import *

def tts_def_two(X, y, val_size=0.2, test_size=0.1):
    X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=test_size+val_size, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val, test_size=(test_size/(test_size+val_size)), random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def tts_def(X, val_size=0.2, test_size=0.1):
    train_data, test_val_data = train_test_split(X, test_size=test_size+val_size, random_state=42)
    val_data, test_data = train_test_split(test_val_data, test_size=(test_size/(test_size+val_size)), random_state=42)
    
    return train_data, val_data, test_data

# Создаетсся датасет, в котором мы берем n вариаций молекул 
# мы применяем train_test_split для вариаций, а не для молекул, чтобы представители всех молекул были во всех выборках
def create_tts_def_mols(df, n = 51, val_size=0.2, test_size=0.1):
    mol_train, mol_val, mol_test = [], [], []
    targets_train, targets_val, targets_test = [], [], []
    train_all, test_all, val_all = [], [], []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        data = []
        for j in range(n):
            xyz_string = row[f'xyz_{j}']
            try:
                y = torch.tensor([row['mouse_intraperitoneal_LD50']])
                data.append((xyz_string_to_graph_data(xyz_string, y), y))
            except RuntimeError:
                print(row)
                continue
        if len(data) > 1:
            train_data, val_data, test_data = tts_def(data, val_size, test_size)
            train_all += train_data
            val_all += val_data
            test_all += test_data
        else:
            train_data += data
    # убираем последовательное добавление каждой молекулы
    random.shuffle(train_all)
    random.shuffle(val_all)
    random.shuffle(test_all)
    
    for t_data in train_all:
        mol_train.append(t_data[0])
        targets_train.append(t_data[0])
        
    for v_data in val_all:
        mol_val.append(v_data[0])
        targets_val.append(v_data[0])
        
    for t_data in test_all:
        mol_test.append(t_data[0])
        targets_test.append(t_data[0])
    
    return mol_train, mol_val, mol_test, targets_train, targets_val, targets_test
        

# Классическое разбиение, где мы берем только одну молекулу
def create_tts_def(df, val_size=0.2, test_size=0.1, n=1):
    mol_train, mol_val, mol_test = [], [], []
    targets_train, targets_val, targets_test = [], [], []
    X = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        data = []
        for j in range(n):
            xyz_string = row[f'xyz_{j}']
            try:
                data.append(xyz_string_to_graph_data(xyz_string, torch.tensor([row['mouse_intraperitoneal_LD50']])))
            except RuntimeError:
                print(f"Error processing row: {row}")
                continue
        X.append(data)
    
    train_data, val_data, test_data = tts_def(X, val_size, test_size)
    
    for i in train_data:
        mol_train += i
        targets_train += [item.y for item in i]
    for i in val_data:
        mol_val += i
        targets_val += [item.y for item in i]
    for i in test_data:
        mol_test += i
        targets_test += [item.y for item in i]
            
    return mol_train, mol_val, mol_test, targets_train, targets_val, targets_test

# В датасете уже было свое разбиение, хотел посмотреть, оно какое-то особенное и имеет смысл или нет
def create_tts_from_df(df, n = 1, all_mol=False):
    mol_train, mol_val, mol_test = [], [], []
    targets_train, targets_val, targets_test = [], [], []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        X = []
        y = []
        if all_mol:
            for j in range(n):
                xyz_string = row[f'xyz_{j}']
                try:
                    data = xyz_string_to_graph_data(xyz_string, torch.tensor([row['mouse_intraperitoneal_LD50']]))
                except RuntimeError:
                    print(row)
                    continue
                X.append(data)
                y.append(data.y)
        else:
            xyz_string = row[f'xyz_0']
            try:
                data = xyz_string_to_graph_data(xyz_string, torch.tensor([row['mouse_intraperitoneal_LD50']]))
            except RuntimeError:
                print(row)
                continue
            X.append(data)
            y.append(data.y)

        if row['split'] == 'train':
            mol_train += X
            targets_train += y
        elif row['split'] == 'val':
            mol_val += X
            targets_val += y
        else:
            mol_test += X
            targets_test += y
        
    return mol_train, mol_val, mol_test, targets_train, targets_val, targets_test


# TODO
