import os
import argparse
import pickle
import torch
from tqdm import tqdm
import numpy as np
import random
import Features.features as features
parser = argparse.ArgumentParser()


def process_fasta(fasta_path):
    """
    Process the input fasta file
    """
    name_seq = {}
    with open(fasta_path, 'r') as r1:
        fasta = r1.readlines() 
        for i in range(len(fasta)):
            if fasta[i][0] == '>':
                name = fasta[i].split('>')[1].replace('\n','')
                seq = fasta[i+1].replace('\n','')
                name_seq[name] = seq
    pickle.dump(name_seq, open('./Data/example.pkl','wb'))
    return name_seq

def extract_features(name_seq, fasta, gpu):
    """
    Extract structural features and ProtTrans embeddings
    """
    ID_list = []
    seq_list = []
    for key in name_seq.keys():
        ID_list.append(key)
        seq_list.append(name_seq[key])
    signal_str = 0
    signal_prot = 0
    signal_dssp = 0
    for id in ID_list:
        if not os.path.exists('./Data/Structures/' + id + '.tensor'):
            signal_str = 1
        if not os.path.exists('./Data/ProtTrans/' + id + '.tensor'):
            signal_prot = 1
        if not os.path.exists('./Data/DSSP/' + id + '.tensor'):
            signal_dssp = 1
    
    if signal_str == 1:
        # get the structural features from ESMFold-predicted structures
        features.get_esmfold(fasta, './Data/Structures/')

    if signal_prot == 1:
        # get the ProtTrans embeddings
        features.get_prottrans(fasta, './Data/ProtTrans/', gpu)
        
    if signal_dssp == 1:
        # get the DSSP features
        dssp_path = "./Features/dssp-2.0.4/"
        features.get_dssp(fasta, dssp_path, './Data/Structures/', './Data/DSSP/')
        


if __name__ == '__main__':
    parser.add_argument("--task", type=str, default='ActiveSite')
    parser.add_argument("--fasta", type=str, default='./Data/fasta/Active_sites.fasta')
    parser.add_argument("--gpu", type=str, default=None)
    args = parser.parse_args()
    name_seq = process_fasta(args.fasta)
    extract_features(name_seq, args.fasta, args.gpu)
    if args.task == 'EC_number':
        # predict the EC numbers
        os.system('python ./Active_sites/main.py')
        os.system('python ./EC_number/main.py --fasta {}'.format(args.fasta))
    elif args.task == 'ActiveSite':
        # predict the active sites
        os.system('python ./Active_sites/main.py')
    elif args.task == 'Optimum_pH':
        # predict the optimum pH
        os.system('python ./Optimum_pH/main.py')
    else:
        print('Please enter the correct task name!')
