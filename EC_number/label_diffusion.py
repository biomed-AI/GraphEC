import scipy.sparse as ssp
from scipy.sparse.linalg import inv
import numpy as np
import os, pickle, time

Diamond_PATH = "./EC_number/tools/"
Dataset_Path = "./EC_number/data/"


def sparse_divide_nonzero(a, b):
    inv_b = b.copy()
    inv_b.data = 1 / inv_b.data
    return a.multiply(inv_b)


def jaccard(W):
    Co = W.dot(W.T)
    CLEN = W.sum(axis=1) # (N, 1) numpy matrix

    # original implementation: J = Co / (CLEN + CLEN.T - Co + 1)
    nonzero_mask = Co.astype('bool')
    denominator = nonzero_mask.multiply(CLEN) + nonzero_mask.multiply(CLEN.T) - Co + nonzero_mask

    J = sparse_divide_nonzero(Co, denominator)
    return J


def compute_L(W):
    JW = jaccard(W).multiply(W)
    degree = 1.0 / JW.sum(axis=1) # shape = (N, 1)
    W_S2F = 0.5 * (JW.multiply(degree) + JW.multiply(degree.T))

    degree = W_S2F.sum(axis=1)
    D_S2F = ssp.spdiags(degree.T, 0, W.shape[0], W.shape[0])
    L_S2F = D_S2F - W_S2F

    return L_S2F


def homology_matrix(fasta, cutoff = 0.1):
    ID = []
    with open(fasta, "r") as f:
        lines = f.readlines()
    for line in lines:
        if line[0] == ">":
            ID.append(line[1:-1])

    ID2idx = dict(zip(ID, range(len(ID))))

    os.system("{}diamond makedb --in {} -d {} --quiet".format(Diamond_PATH, fasta, fasta))
    print("Start homology alignment...")
    os.system('{}diamond blastp -d {}.dmnd -q {} -o {} --very-sensitive --quiet -p 8'.format(Diamond_PATH, fasta, fasta, fasta + ".tsv"))
    print("Alignment done.")

    with open(fasta + ".tsv", "r") as f:
        out = f.readlines()

    os.system("rm {}.*".format(fasta))

    homology = {}
    for line in out:
        fields = line.strip().split()
        query_idx = ID2idx[fields[0]]
        subject_idx = ID2idx[fields[1]]
        identity = float(fields[2]) / 100

        if query_idx not in homology:
            homology[query_idx] = {}
        if subject_idx in homology[query_idx]:
            homology[query_idx][subject_idx] = max(homology[query_idx][subject_idx], identity)
        else:
            homology[query_idx][subject_idx] = identity

        # handle symmetry
        if subject_idx not in homology:
            homology[subject_idx] = {}
        if query_idx in homology[subject_idx]:
            homology[subject_idx][query_idx] = max(homology[subject_idx][query_idx], identity)
        else:
            homology[subject_idx][query_idx] = identity

    row = []
    col = []
    data = []
    for i in homology:
        for j in homology[i]:
            val = homology[i][j]

            if i == j:
                val = 1.0

            if val < cutoff:
                continue

            row.append(i)
            col.append(j)
            data.append(val)

    data = np.array(data)
    graph = ssp.csr_matrix((data, (row, col)), shape = (len(ID), len(ID)))

    return graph


def LabelDiffusion(initial_pred, lamda, args, identity_cutoff = 0.1):
    '''
    This is an efficient implementation of the label diffusion algorithm
    by S2F, using the sparse matrix throughout the caculation.
    W: homology sparse matrix
    '''
    train_fasta = Dataset_Path + "EC_train.fa"
    test_fasta = args.fasta

    os.system("{}diamond makedb --in {} -d {} --quiet".format(Diamond_PATH, train_fasta, train_fasta))
    print("Start alignment to the training set...")
    os.system('{}diamond blastp -d {}.dmnd -q {} -o {} --very-sensitive --quiet -p 8'.format(Diamond_PATH, train_fasta, test_fasta, "testVStrain.tsv"))
    os.system("rm {}.dmnd".format(train_fasta))
    print("Alignment done.")


    with open("testVStrain.tsv", "r") as f:
        testVStrain = f.readlines()

    train_seed_ID = set()
    for line in testVStrain:
        fields = line.strip().split()
        subject_id = fields[1]
        identity = float(fields[2]) / 100

        if identity > identity_cutoff:
            train_seed_ID.add(subject_id)

    os.system("rm testVStrain.tsv")

    with open(Dataset_Path + "EC_train.pkl", "rb") as f:
        train_dataset = pickle.load(f)

    train_seed_ID_list = []
    train_seed_fasta = ""
    for ID in train_dataset: # keep the original ID order
        if ID in train_seed_ID:
            train_seed_ID_list.append(ID)
            train_seed_fasta += (">" + ID + "\n" + train_dataset[ID][0] + "\n")

    with open("train_seed_seq.fa", "w") as f:
        f.write(train_seed_fasta)

    os.system("cat train_seed_seq.fa {} > train_seed_and_test.fa".format(test_fasta))
    os.system("rm train_seed_seq.fa")


    W = homology_matrix("train_seed_and_test.fa")
    os.system("rm train_seed_and_test.fa")


    # get train seed label
    row = []
    col = []
    for i, ID in enumerate(train_seed_ID_list):
        label_idx = train_dataset[ID][1]
        row += [i] * len(label_idx)
        col += label_idx
    data = [1] * len(row)

    label_size = 5106
    train_seed_label = ssp.csr_matrix((data, (row, col)), shape = (len(train_seed_ID_list), label_size))


    # Label Diffusion
    print("Start diffusion...")
    start = time.time()

    initial_pred = ssp.csr_matrix(initial_pred)
    initial_pred = ssp.vstack([train_seed_label, initial_pred])

    L = compute_L(W)

    if type(lamda) == list:
        diffusion_pred = []
        for lamda_i in lamda:
            IlambdaL = ssp.identity(W.shape[0]) + L.multiply(lamda_i)
            kernel = inv(IlambdaL.tocsc())[len(train_seed_ID_list):]
            diffusion_pred.append(kernel.dot(initial_pred).toarray())
    else:
        IlambdaL = ssp.identity(W.shape[0]) + L.multiply(lamda)
        kernel = inv(IlambdaL.tocsc())[len(train_seed_ID_list):]
        diffusion_pred = kernel.dot(initial_pred).toarray()

    end = time.time()
    print("Diffusion done! Cost {}s.".format(end - start))

    return diffusion_pred
