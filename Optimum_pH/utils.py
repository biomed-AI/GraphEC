import numpy as np
import os, random, pickle
import datetime
from tqdm import tqdm
from sklearn.metrics import auc, roc_auc_score, precision_recall_curve, mean_squared_error, matthews_corrcoef,r2_score, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
from sklearn import metrics
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
    'augment_eps': 0.15,
    'dropout': 0.2,
    'task':'Sol',
    'batch_size': 16,
    'folds': 5,
    'r':15,
    'num_workers':8,
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
    predict the optimum pH
    """
    device = torch.device('cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu')

    node_input_dim = config['node_input_dim']
    edge_input_dim = config['edge_input_dim']
    hidden_dim = config['hidden_dim']
    layer = config['layer']
    augment_eps = config['augment_eps']
    dropout = config['dropout']
    num_workers = config['num_workers']
    folds = config['folds']
    r = config['r']
    task = config['task']
    
    # load the data for prediction
    with open('./Data/' + "example.pkl", "rb") as f:
        test_data = pickle.load(f)

    # construct the dataset
    test_dataset = ProteinGraphDataset(test_data, range(len(test_data)), args,r)
    test_dataloader = DataLoader(test_dataset, batch_size = 16, shuffle=False, drop_last=False, num_workers=num_workers, prefetch_factor=2)

    models = []
    for fold in range(folds):
        # load the model
        state_dict = torch.load('./Optimum_pH/model/' + 'fold%s.ckpt'%fold, device)

        # define the model
        model = model_class(node_input_dim, edge_input_dim, hidden_dim, layer, dropout, augment_eps, task,device).to(device)
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
            outputs = [model(data.X, data.node_feat, data.edge_index, data.seq, data.batch) for model in models]

            # obtain the mean of the prediction results
            outputs = torch.stack(outputs,0).mean(0) 
        
        test_pred += list(outputs.detach().cpu().numpy())

        IDs = data.name
        for i, ID in enumerate(IDs):
            test_pred_dict[ID] = outputs[i*3:(i+1)*3].cpu()
    pred_list = ['acidic', 'neutral', 'alkaline']
    for key in test_pred_dict.keys():
        with open('./Optimum_pH/results/{name}.txt'.format(name=key),'w') as w1:
            w1.writelines('The results of GraphEC-pH' + '\n')
            w1.writelines('Name' + '\t' + 'prediction' + '\n')
            print(pred_list[torch.argmax(test_pred_dict[key])])
            w1.writelines(key + '\t' + pred_list[torch.argmax(test_pred_dict[key])] + '\n')
