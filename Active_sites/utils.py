import numpy as np
import os, random, pickle
import datetime
from tqdm import tqdm
from sklearn.metrics import auc, roc_auc_score, precision_recall_curve
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
from torch.utils.data import RandomSampler
import torch_geometric
from torch_geometric.loader import DataLoader
from data import *



config = {
    'node_input_dim': 1024 + 9 + 184, 
    'edge_input_dim': 450,
    'hidden_dim': 128,
    'layer': 4,
    'augment_eps': 0.1,
    'dropout': 0.2,
    'batch_size': 1,
    'folds': 5,
    'num_workers': 8,
}

def Seed_everything(seed=42):
    """
    define the random Seed
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def predict(model_class, args):
    """
    predict the active sites
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    node_input_dim = config['node_input_dim']
    edge_input_dim = config['edge_input_dim']
    hidden_dim = config['hidden_dim']
    layer = config['layer']
    augment_eps = config['augment_eps']
    dropout = config['dropout']
    num_workers = config['num_workers']
    batch_size = config['batch_size']
    folds = config['folds']

    # load the data for prediction
    with open('./Data/' + "example.pkl", "rb") as f:
        test_data = pickle.load(f)
    
    # construct the dataset
    test_dataset = ProteinGraphDataset(test_data, range(len(test_data)), args)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False, drop_last=False, num_workers=num_workers, prefetch_factor=2)

    models = []
    for fold in range(folds):
        # load the model
        state_dict = torch.load('./Active_sites/model/' + 'fold%s.ckpt'%fold, device)
        # define the model
        model = model_class(node_input_dim, edge_input_dim, hidden_dim, layer, dropout, augment_eps, task='ActiveSite').to(device)
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)

    test_pred_dict = {}
    test_pred = []
    test_y = []
    for data in tqdm(test_dataloader):
        data = data.to(device)

        with torch.no_grad():
            # get the predictions
            outputs = [model(data.X, data.node_feat, data.edge_index, data.seq, data.batch).sigmoid() for model in models]
            # obtain the mean of the prediction results
            outputs = torch.stack(outputs,0).mean(0)

        test_pred += list(outputs.detach().cpu().numpy())
       
        # export prediction results
        IDs = data.name
        outputs_split = torch_geometric.utils.unbatch(outputs, data.batch)
        for i, ID in enumerate(IDs):
            test_pred_dict[ID] = []
            test_pred_dict[ID].append(outputs_split[i].detach().cpu())

    for key in test_pred_dict.keys():
        with open('./Active_sites/results/{name}.txt'.format(name=key),'w') as w1:
            w1.writelines('The results of GraphEC-AS' + '\n')
            w1.writelines('Num' + '\t' + 'AA' + '\t' + 'Score' + '\n')
            for i in range(len(test_pred_dict[key][0])):
                w1.writelines(str(i) + '\t' + str(test_data[key][i]) + '\t' + str(float(test_pred_dict[key][0][i][0])) + '\n')
    with open('./Active_sites/results/' + "test_pred_dict.pkl", "wb") as f:
        pickle.dump(test_pred_dict, f)

                    


               

