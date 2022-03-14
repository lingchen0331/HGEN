# -*- coding: utf-8 -*-

import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from walker import load_a_HIN_from_pandas
from model import NSTrainSet, HIN2vec, train
import pickle
import csv
# set method parameters
window = 5
walk = 8
walk_length = 10
embed_size = 32
neg = 8
sigmoid_reg = True 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'device = {device}')

directory = "pubmed_new"

# set dataset [PLEASE USE YOUR OWN DATASET TO REPLACE THIS]
demo_edge = pd.read_csv('./{}.csv'.format(directory), index_col=0)

edges = [demo_edge]

print('finish loading edges')

# init HIN
hin = load_a_HIN_from_pandas(edges)
hin.window = window

dataset = NSTrainSet(hin.sample(walk_length, walk), hin.node_size, neg=neg)

hin2vec = HIN2vec(hin.node_size, hin.path_size, embed_size, sigmoid_reg).to(device)

# set training parameters
n_epoch = 4
batch_size = 32
log_interval = 200

data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
optimizer = optim.AdamW(hin2vec.parameters())
loss_function = nn.BCELoss()

for epoch in range(n_epoch):
    train(log_interval, hin2vec, device, data_loader, optimizer, loss_function, epoch)

#torch.save(hin2vec, 'hin2vec.pt')

# set output parameters [the output file is a bit different from the original code.]
node_vec_fname = 'hin2vec_{}_32.p'.format(directory)
# path_vec_fname = 'meta_path_vec.txt'
path_vec_fname = None

print(f'saving node embedding vectors to {node_vec_fname}...')
node_embeds = pd.DataFrame(hin2vec.start_embeds.weight.data.cpu().numpy())
node_embeds = node_embeds.rename(hin.id2node)

node_embedding = {}
indices = list(node_embeds.index)
for i, x in enumerate(indices):
    node_embedding[x] = list(node_embeds.iloc[i])
    
pickle.dump(node_embedding, open(node_vec_fname, "wb"))
'''
if path_vec_fname:
    print(f'saving meta path embedding vectors to {path_vec_fname}...')
    path_embeds = pd.DataFrame(hin2vec.path_embeds.weight.data.numpy())
    path_embeds.rename(hin.id2path).to_csv(path_vec_fname, sep=' ')
       
# save model
# torch.save(hin2vec.state_dict(), 'hin2vec.pt')
'''

