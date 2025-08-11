import torch
from transformers import T5EncoderModel, T5Tokenizer
import re, argparse
import numpy as np
from tqdm import tqdm
import gc
import multiprocessing
import os, datetime
from Bio import pairwise2
import pickle


"""
get the ProtTrans embeddings and structural features
"""

dssp_path = "./Features/dssp-2.0.4/"

def get_prottrans(fasta_file,output_path, gpu):
    """
    get ProtTrans embeddings
    """
    num_cores = 2
    multiprocessing.set_start_method("forkserver")
    os.environ["OMP_NUM_THREADS"] = str(num_cores)
    ID_list = []
    seq_list = []
    with open(fasta_file, "r") as f:
        lines = f.readlines()
    for line in lines:
            if line[0] == ">":
                ID_list.append(line[1:-1])
            else:
                seq_list.append(" ".join(list(line.strip())))

    # Replace it with your own path
    Max_protTrans_path = "../Data/ProtTrans/Max_protTrans.npy"
    Min_protTrans_path = "../Data/ProtTrans/Min_protTrans.npy"
    model_path = "/home/songyd/software/Prot-T5-XL-U50"
    tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_path)
    Max_protTrans = np.load(open(Max_protTrans_path, 'rb'))
    Min_protTrans = np.load(open(Min_protTrans_path, 'rb'))
    gc.collect()
    device = torch.device('cuda:' + gpu if torch.cuda.is_available() and gpu else 'cpu')
    model = model.eval()
    model = model.cuda()
    print(next(model.parameters()).device)
    print('starttime')
    starttime = datetime.datetime.now()
    print(starttime)
    batch_size = 1

    for i in tqdm(range(0, len(ID_list), batch_size)):
        if i + batch_size <= len(ID_list):
            batch_ID_list = ID_list[i:i + batch_size]
            batch_seq_list = seq_list[i:i + batch_size]
        else:
            batch_ID_list = ID_list[i:]
            batch_seq_list = seq_list[i:]
        

        # Create or load sequences and map rarely occured amino acids (U,Z,O,B) to (X)
        batch_seq_list = [re.sub(r"[UZOB]", "X", sequence) for sequence in batch_seq_list]

        # Tokenize, encode sequences and load it into the GPU if possibile
        ids = tokenizer.batch_encode_plus(batch_seq_list, add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)

        # Extracting sequences' features and load it into the CPU if needed
        with torch.no_grad():
            embedding = model(input_ids=input_ids,attention_mask=attention_mask)
        embedding = embedding.last_hidden_state.cpu().numpy()
        

        # Remove padding (\<pad>) and special tokens (\</s>) that is added by ProtT5-XL-UniRef50 model
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][:seq_len-1]
            seq_emd = (seq_emd - Min_protTrans) / (Max_protTrans - Min_protTrans)
            torch.save(seq_emd, output_path + batch_ID_list[seq_num] + '.tensor')
            endtime = datetime.datetime.now()
            print('endtime')
            print(endtime)

def get_pdb_xyz(pdb_file):
    """
    get the coordinates
    """
    current_pos = -1000
    X = []
    current_aa = {} # N, CA, C, O, R
    for line in pdb_file:
        if (line[0:4].strip() == "ATOM" and int(line[22:26].strip()) != current_pos) or line[0:4].strip() == "TER":
            if current_aa != {}:
                R_group = []
                for atom in current_aa:
                    if atom not in ["N", "CA", "C", "O"]:
                        R_group.append(current_aa[atom])
                if R_group == []:
                    R_group = [current_aa["CA"]]
                R_group = np.array(R_group).mean(0)
                X.append([current_aa["N"], current_aa["CA"], current_aa["C"], current_aa["O"], R_group])
                current_aa = {}
            if line[0:4].strip() != "TER":
                current_pos = int(line[22:26].strip())

        if line[0:4].strip() == "ATOM":
            atom = line[13:16].strip()
            if atom != "H":
                xyz = np.array([line[30:38].strip(), line[38:46].strip(), line[46:54].strip()]).astype(np.float32)
                current_aa[atom] = xyz
    return np.array(X)

def get_esmfold(fasta_file,output_path):
    os.system('python ./Features/esmfold/esmfold.py -i {fasta} -o {output} --chunk-size 128'.format(fasta=fasta_file,output=output_path))
    pdbfasta = {}
    with open(fasta_file) as r1:
        fasta_ori = r1.readlines()
    for i in range(len(fasta_ori)):
        if fasta_ori[i][0] == ">":
            name = fasta_ori[i].split('>')[1].replace('\n','')
            seq = fasta_ori[i+1].replace('/n','')
            pdbfasta[name] = seq
    for key in pdbfasta.keys():
        with open(output_path + key + '.pdb','r') as r1:
            pdb_file = r1.readlines()
        coord = get_pdb_xyz(pdb_file)
        torch.save(torch.tensor(coord, dtype = torch.float32), output_path + key + '.tensor')

def process_dssp(dssp_file):
    aa_type = "ACDEFGHIKLMNPQRSTVWY"
    SS_type = "HBEGITSC"
    rASA_std = [115, 135, 150, 190, 210, 75, 195, 175, 200, 170,
                185, 160, 145, 180, 225, 115, 140, 155, 255, 230]

    with open(dssp_file, "r") as f:
        lines = f.readlines()

    seq = ""
    dssp_feature = []

    p = 0
    while lines[p].strip()[0] != "#":
        p += 1
    for i in range(p + 1, len(lines)):
        aa = lines[i][13]
        if aa == "!" or aa == "*":
            continue
        seq += aa
        SS = lines[i][16]
        if SS == " ":
            SS = "C"
        SS_vec = np.zeros(8)
        SS_vec[SS_type.find(SS)] = 1
        ACC = float(lines[i][34:38].strip())
        ASA = min(1, ACC / rASA_std[aa_type.find(aa)])
        dssp_feature.append(np.concatenate((np.array([ASA]), SS_vec)))

    return seq, dssp_feature


def match_dssp(seq, dssp, ref_seq):
    alignments = pairwise2.align.globalxx(ref_seq, seq)
    ref_seq = alignments[0].seqA
    seq = alignments[0].seqB

    padded_item = np.zeros(9)

    new_dssp = []
    for aa in seq:
        if aa == "-":
            new_dssp.append(padded_item)
        else:
            new_dssp.append(dssp.pop(0))

    matched_dssp = []
    for i in range(len(ref_seq)):
        if ref_seq[i] == "-":
            continue
        matched_dssp.append(new_dssp[i])

    return matched_dssp


def get_dssp(fasta, dssp_path, esmfold_path, dssp_save):
    """
    get dssp features
    """
    with open(fasta, 'r') as r1:
        data_fasta = r1.readlines()
    for i in range(0,len(data_fasta)-1,2):
        ID = data_fasta[i].split('>')[1].replace('\n','')
        ref_seq = data_fasta[i+1].replace('\n','')
        os.system("{}mkdssp -i {}{}.pdb -o {}{}.dssp".format(dssp_path, esmfold_path, ID, dssp_save, ID))
        dssp_seq, dssp_matrix = process_dssp("{}{}.dssp".format(dssp_save,ID))
        if dssp_seq != ref_seq:
            dssp_matrix = match_dssp(dssp_seq, dssp_matrix, ref_seq)
        torch.save(torch.tensor(dssp_matrix, dtype = torch.float32), "{}{}.tensor".format(dssp_save, ID))
        os.system("rm {}{}.dssp".format(dssp_save, ID))
