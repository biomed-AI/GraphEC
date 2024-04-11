import pickle
import numpy as np
import os, random, datetime
from tqdm import tqdm
import warnings
warnings.simplefilter('ignore')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.utils.data import RandomSampler
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, auc, precision_recall_curve, roc_auc_score
from sklearn.model_selection import KFold
from model import *
from label_diffusion import *
from data import *


NN_config = {
    'model_class': GraphEC,
    'feature_dim': 1024+9+184,
    'edge_input_dim':450,
    'hidden_dim': 256,
    'layer': 3,
    'num_heads': 8,
    'augment_eps': 0,
    'dropout': 0.1,
    'label_szie':5106,
    'batch_size': 32,
    'device':6,
    'num_workers':64,
}

def Seed_everything(seed=2022):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  
        
def padding_ver1(x, batch_id, feature_dim, activate_site):
    batch_size = max(batch_id) + 1
    max_len= max(torch.unique(batch_id,return_counts=True)[1])
    batch_data = torch.zeros([batch_size,max_len,feature_dim])
    mask = torch.zeros([batch_size,max_len])
    batch_activate_site = torch.zeros([batch_size, max_len, 1])
    len_0 = 0
    len_1 = 0
    for i in range(batch_size):
        len_1 = len_0 + torch.unique(batch_id,return_counts=True)[1][i]
        batch_data[i][:torch.unique(batch_id,return_counts=True)[1][i]] = x[len_0:len_1]
        batch_activate_site[i][:torch.unique(batch_id,return_counts=True)[1][i]] = activate_site[len_0:len_1]
        mask[i][:torch.unique(batch_id,return_counts=True)[1][i]] = 1
        len_0 += torch.unique(batch_id,return_counts=True)[1][i]
    return batch_data, mask, batch_activate_site

def predict(args=None, seed=None):
    model_class = NN_config['model_class']
    node_input_dim = NN_config['feature_dim']
    edge_input_dim = NN_config['edge_input_dim']
    hidden_dim = NN_config['hidden_dim']
    layer = NN_config['layer']
    augment_eps = NN_config['augment_eps']
    dropout = NN_config['dropout']
    batch_size = NN_config['batch_size']
    num_workers = NN_config['num_workers']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open("./Active_sites/results/test_pred_dict.pkl",'rb') as r1:
        test_active_sites = pickle.load(r1)

    with open('./Data/example.pkl', "rb") as f:
        test_data = pickle.load(f)
    test_dataset = ProteinGraphDataset(test_data, range(len(test_data)), args, test_active_sites)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, prefetch_factor=2)

    models = []
    for fold in range(5):
        state_dict = torch.load('./EC_number/model/' + 'fold%s.ckpt'%fold, device)
        model = model_class(node_input_dim, edge_input_dim, hidden_dim, layer, dropout, augment_eps, device).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)
    print('model count:',len(models))

    test_pred = []
    name = []
    for batch in tqdm(test_dataloader):
        with torch.no_grad():
            batch_data, mask_data, batch_activate_site = padding_ver1(batch.node_feat, batch.batch, batch.node_feat.shape[1], batch.activate_site)
            batch.to(device)
            batch_activate_site = batch_activate_site.to(device)
            preds = [model(batch.X, batch.node_feat, batch.edge_index, batch.seq, batch.batch, batch_data, mask_data, batch_activate_site) for model in models]
            name.extend(batch.name)
            preds = torch.stack(preds,0).mean(0) 
            test_pred.append(preds.sigmoid().detach().cpu().numpy())

    test_pred = np.concatenate(test_pred)

    # Label diffusion
    lamda_list = [0, 0.1, 0.5, 1]
    diffusion_pred = LabelDiffusion(test_pred, lamda_list)
    EC_id = pickle.load(open('./EC_number/data/EC_idx.pkl','rb'))
    id_EC = dict([val, key] for key, val in EC_id.items())
    w2 = open('./EC_number/results/example_top5.txt','w')
    with open('./EC_number/results/example_all.txt','w') as w1:
        w1.writelines("The results of GraphEC" + '\n')
        for i in range(len(id_EC)):
            w1.writelines(id_EC[i] + ',' + ' ')
        w1.writelines('\n')
        w2.writelines("The results of GraphEC" + '\n')
      
        for i in range(len(lamda_list)):
            test_pred = diffusion_pred[i]
            if i == 1:
                for j  in range(len(name)):
                    w1.writelines(name[j] + '\n')
                    top_5 = np.argsort(np.array(test_pred[j]))[-5:][::-1]
                    for k in range(len(test_pred[j])):
                        w1.writelines(str(round(test_pred[j][k],3)) + ',' + ' ')
                    w1.writelines('\n')
                    w2.writelines(name[j] + '\n')
                    for id in top_5:
                        w2.writelines('EC: ' + id_EC[id] + ' | ' + str(round(test_pred[j][id],3)) + '\n')
                        
                    
                    