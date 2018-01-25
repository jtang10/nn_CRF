from __future__ import print_function, division
import time
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

class Protein_Dataset(Dataset):
    def __init__(self, relative_path, datalist_addr, max_seq_len=300, padding=True):
        self.relative_path = relative_path
        self.protein_list = self.read_list(relative_path + datalist_addr)
        self.max_seq_len = max_seq_len
        self.padding = padding
        self.dict_ss = {key: value for (key, value) in \
            zip(['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T'], range(8))}

    def __len__(self):
        return len(self.protein_list)

    def __getitem__(self, idx):
        protein_name = self.protein_list[idx]
        features, SS8, length, ACC = self.read_protein(protein_name, self.relative_path, self.max_seq_len, self.padding)
        return torch.from_numpy(features).float(), torch.from_numpy(SS8).long(), length, torch.from_numpy(ACC).long()

    def read_list(self, filename):
        """Given the filename storing all protein names, return a list of protein names.
        """
        with open(filename) as f:
            proteins_names = f.read().splitlines()
        return proteins_names

    def read_protein(self, protein_name, relative_path, max_seq_len=300, padding=False):
        """Given a protein name, return the ndarray of features [1 x seq_len x n_features]
        and labels [1 x seq_len].
        """
        features_addr = relative_path + '66FEAT/' + protein_name + '.66feat'
        SS8_addr = relative_path + 'Angles/' + protein_name + '.ang'
        ACC_addr = relative_path + 'TPL/' + protein_name + '.tpl'

        protein_features = np.loadtxt(features_addr)
        SS8 = []
        ACC = []
        # Reading SS8 label
        with open(SS8_addr) as f:
            next(f)
            for i, line in enumerate(f):
                line = line.split('\t')
                if line[0] == '0':
                    # 0 means the current ss label exists.
                    SS8.append(self.dict_ss[line[3]])
        SS8 = np.array(SS8)
        protein_length = SS8.shape[0]
        # Reading solvent accessibility label
        with open(ACC_addr) as f:
            for _ in range(12):
                next(f)
            for i, line in enumerate(f):
                if line == '\n':
                    break
                line = line.split()
                if line[2] == '0':
                    ACC.append(int(line[5]))
        ACC = np.array(ACC)

        if padding:
            # if features passes max_seq_len, cutoff
            if protein_features.shape[0] >= max_seq_len:
                protein_features = protein_features[:max_seq_len, :]
                SS8 = SS8[:max_seq_len]
                protein_length = max_seq_len
            # else, zero-pad to max_seq_len
            else:
                padding_length = max_seq_len - protein_features.shape[0]
                protein_features = np.pad(protein_features, ((0, padding_length), (0, 0)),
                                          'constant', constant_values=((0, 0), (0, 0)))
                SS8 = np.pad(SS8, (0, padding_length), 'constant', constant_values=(0, 0))
                ACC = np.pad(ACC, (0, padding_length), 'constant', constant_values=(0, 0))
        return protein_features, SS8, protein_length, ACC

if __name__ == '__main__':
    SetOf7604Proteins_path = '../data/ProteinProperty_Project/SetOf7604Proteins/'
    CASP11_path = '../data/ProteinProperty_Project/CASP11/'
    trainList_addr = 'trainList'
    validList_addr = 'validList'
    testList_addr = 'testList'

    protein_dataset = Protein_Dataset(CASP11_path, 'proteinList', max_seq_len=698, padding=False)
    dataloader = DataLoader(protein_dataset, batch_size=1, shuffle=False, num_workers=1)
    print(protein_dataset.protein_list[0])  
    for i, sample_batched in enumerate(dataloader):
        features, SS8, lengths, ACC = sample_batched
        print(features.size())
        print(SS8.size())
        print(ACC.size())
        break
