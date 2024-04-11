import os
import argparse
import pickle
import torch
from tqdm import tqdm
import numpy as np
import random
from feature_extraction.ProtTrans import get_ProtTrans
from feature_extraction.process_structure import get_pdb_xyz, process_dssp, match_dssp
parser = argparse.ArgumentParser()


def process_fasta(fasta_path):
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
        max_len = max([len(seq) for seq in seq_list])
        chunk_size = 32 if max_len > 1000 else 64
        esmfold_cmd = "python ./feature_extraction/esmfold.py -i {} -o {} --chunk-size {}".format(fasta, './Data/Structures/', chunk_size)
        if not gpu:
            esmfold_cmd += " --cpu-only"
        else:
            esmfold_cmd = "CUDA_VISIBLE_DEVICES=" + gpu + " " + esmfold_cmd
        os.system(esmfold_cmd + " | tee ./esmfold_pred.log")
        for ID in tqdm(ID_list):
            with open('./Data/Structures/' + ID + ".pdb", "r") as f:
                X = get_pdb_xyz(f.readlines())
            torch.save(torch.tensor(X, dtype = torch.float32), './Data/Structures/' + ID + '.tensor')
    if signal_prot == 1:
        # Replace with your own path
        ProtTrans_path = "/data/user/songyd/software/Prot-T5-XL-U50"

        outpath = './Data/ProtTrans/'
        get_ProtTrans(ID_list, seq_list, ProtTrans_path, outpath, gpu)
    if signal_dssp == 1:
        for i in tqdm(range(len(ID_list))):
            ID = ID_list[i]
            seq = seq_list[i]
            os.system("./feature_extraction/mkdssp -i ./Data/Structures/{}.pdb -o ./Data/DSSP/{}.dssp".format(ID, ID))
            dssp_seq, dssp_matrix = process_dssp("./Data/DSSP/{}.dssp".format(ID))
            if dssp_seq != seq:
                dssp_matrix = match_dssp(dssp_seq, dssp_matrix, seq)
            torch.save(torch.tensor(np.array(dssp_matrix), dtype = torch.float32), "./github/Data/DSSP/{}.tensor".format(ID))
            os.system("rm ./Data/DSSP/{}.dssp".format(ID))
        



if __name__ == '__main__':
    parser.add_argument("--task", type=str, default='ActiveSite')
    parser.add_argument("--fasta", type=str, default='./Data/fasta/Active_sites.fasta')
    parser.add_argument("--gpu", type=str, default=None)
    args = parser.parse_args()
    name_seq = process_fasta(args.fasta)
    extract_features(name_seq, args.fasta, args.gpu)

    if args.task == 'EC_number':
        os.system('python ./Active_sites/main.py')
        os.system('python ./EC_number/main.py')
    elif args.task == 'ActiveSite':
        os.system('python ./Active_sites/main.py')
    elif args.task == 'Optimum_pH':
        os.system('python ./Optimum_pH/main.py')
    else:
        print('Please enter the correct task name!')