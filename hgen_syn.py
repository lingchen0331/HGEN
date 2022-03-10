# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import pickle
import utils
import scipy.sparse as sp
import networkx as nx
from collections import Counter
from torch_two_sample import MMDStatistic
import matplotlib.pyplot as plt
import time

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--W-down-generator-size', type=int, default=128)
parser.add_argument('--W-down-discriminator-size', type=int, default=128)
parser.add_argument("--batch-size", default=32,
                    help="Batch size of the training/testing set")
parser.add_argument("--noise-dim", default=64,
                    help="The dim of the random noise that is used as input.")
parser.add_argument("--noise-type", choices=["Gaussian", "Uniform"], default="Uniform",
                    help="The noise type to feed into the generator.")
parser.add_argument("--hidden-units", default=128,
                    help="The dimension of the hidden unit in lstm cells.")
parser.add_argument("--num-G-layer", type=int, default=10,
                    help="The number of layers in lstm cells of Generator.")
parser.add_argument("--max-path-len", type=int, default=4,
                    help="The maximum meta path length")
parser.add_argument("--neighbor-sampling-size", type=int, default=120,
                    help="The sampling size from the neighbor nodes")
parser.add_argument("--lr_gen", type=float, default=1e-4,
                    help="Learning Rate of generator")
parser.add_argument("--lr_dis", type=float, default=1e-4,
                    help="Learning Rate of descriminator")
parser.add_argument("--disc-iters", type=int, default=5,
                    help="Discriminator update iterations")
parser.add_argument("--clip-value", type=float, default=0.03,
                    help="lower and upper clip value for disc. weights")
parser.add_argument("--n-critic", type=int, default=5,
                    help="number of training steps for discriminator per iter")
parser.add_argument("--n-epochs", type=int, default=3,
                    help="The number of epochs.")
parser.add_argument("--dataset", choices=["syn_100", "syn_200", "syn_500", "PubMed", "IMDB_movie", "DBLP_four_area"], 
                    default="syn_100", help="The choice of dataset.")
parser.add_argument("--edge-type", type=bool, default=False,
                    help="Whether to predict edge type.")
parser.add_argument("--save", type=bool, default=False,
                    help="Whether to save the trained model.")
args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

val_share = 0.0
test_share = 0.35
seed = 481516234

def data_preprocess(val_share, test_share, seed, directory=args.dataset):
    _X_obs, _A_obs = utils.load_HIN(directory)
    _A_obs[_A_obs > 1] = 1
    _A_obs[_A_obs < 0] = 0
    diag_A = np.diag(np.diag(np.ones(_A_obs.shape)))
    _A_obs = _A_obs - sp.csr_matrix(diag_A)
    _A_obs = _A_obs + _A_obs.T
    _A_obs[_A_obs > 1] = 1
    _A_obs[_A_obs < 0] = 0
    lcc = utils.largest_connected_components(_A_obs)
    _A_obs = _A_obs[lcc, :][:, lcc]
    _N = _A_obs.shape[0]
    _X_obs = dict(zip(lcc, [_X_obs[i] for i in lcc]))
    _X_obs = dict(zip(list(range(len(_X_obs))), _X_obs.values()))
    node_type = utils.group_dict(_X_obs)
    # Spliting training graph and testing graph
    train_ones, val_ones, val_zeros, test_ones, test_zeros = utils.train_val_test_split_adjacency(
            _A_obs, val_share, test_share, seed, undirected=True, connected=False, asserts=False)
    test_graph = sp.coo_matrix((np.ones(len(test_ones)),(test_ones[:,0], test_ones[:,1]))).tocsr()

    return _A_obs, _X_obs, _N, lcc, node_type, test_graph


def read_node_embeddings(node_type, lcc, device, dataset=args.dataset):
    #node_embs = torch.load(dataset+"/final_emb.pt")
    node_embs = pickle.load(open(dataset+"/hin2vec_{}_32.p".format(dataset),"rb"))
    node_embs_classified = []
    bb = dict(zip(list(range(len(lcc))), lcc))
    for i in range(len(node_type)):
        temp = []
        for j in node_type[i]:
            temp.append(torch.tensor(node_embs[bb[j]]).unsqueeze(0))
        node_embs_classified.append(torch.cat(temp, 0).to(device))
    return node_embs_classified


class HINDataset(torch.utils.data.Dataset):
    def __init__(self, node_attribute, num_nodes, max_path_len, dataset):
        self.node_attribute = node_attribute
        self.num_nodes = num_nodes
        self.max_path_len = max_path_len
        self.dataset = dataset
        if self.dataset in ["syn_100", "syn_200", "syn_500"]:
            self.num_classes = 3+1
        else:
            self.num_classes = 4+1
        self.data = self.cache(self.node_attribute, self.num_nodes)

    def cache(self, node_attribute, num_nodes):
        path_len_2 = pickle.load(open(self.dataset+"/path_len_2.p", "rb"))
        path_len_3 = pickle.load(open(self.dataset+"/path_len_3.p", "rb"))
        path_len_4 = pickle.load(open(self.dataset+"/path_len_4.p", "rb"))
        paths = np.array(path_len_2 + path_len_3 + path_len_4)
        g = np.random.Generator(np.random.PCG64())
        sampled_paths = paths[g.choice(len(paths), 6400, replace=False)]
        real_walk_data = []
        for path in sampled_paths:
            temp_walk = utils.one_hot_encoder(np.array(path), num_nodes+1)
            temp_type = utils.one_hot_encoder(np.array([node_attribute[i] for i in path]), self.num_classes)
            # Padding random walks to the max length
            if temp_type.shape[0] < self.max_path_len:
                temp_walk = utils.pad_along_axis(temp_walk, self.max_path_len, axis=0)
                temp_type = utils.pad_along_axis(temp_type, self.max_path_len, axis=0)
            real_walk_data.append((temp_type, temp_walk))
        print("Done!")
        return real_walk_data

    def __getitem__(self, item):
        types, walks = self.data[item]
        return types, walks

    def __len__(self):
        return len(self.data)


class Generator(nn.Module):
    def __init__(self, node_type, N, node_embs, device, args):
        super(Generator, self).__init__()
        self.node_type = node_type
        self.N = N
        self.batch_size = args.batch_size
        self.hidden_units = args.hidden_units
        self.noise_dim = args.noise_dim
        self.max_path_len = args.max_path_len
        self.W_down_generator_size = args.W_down_generator_size
        self.num_G_layer = args.num_G_layer
        self.node_embs = node_embs
        self.device = device
        self.node_emb_size = 32
        
        self.type_0 = torch.tensor(node_type[0], dtype=torch.float).to(self.device)
        self.type_1 = torch.tensor(node_type[1], dtype=torch.float).to(self.device)
        self.type_2 = torch.tensor(node_type[2], dtype=torch.float).to(self.device)
        
        self.lin_node_0 = nn.Linear(self.hidden_units, self.node_emb_size)
        self.lin_node_1 = nn.Linear(self.hidden_units, self.node_emb_size)
        self.lin_node_2 = nn.Linear(self.hidden_units, self.node_emb_size)
        
        if args.dataset in ["syn_100", "syn_200", "syn_500"]:
            self.node_classes = 4
        else:
            self.node_classes = 5
            self.type_3 = torch.tensor(node_type[3], dtype=torch.float).to(self.device)
            self.lin_node_3 = nn.Linear(self.hidden_units, self.node_emb_size)
            
        self.W_down_generator_type = nn.Linear(self.node_classes, self.W_down_generator_size)
        self.W_down_generator_node = nn.Linear(self.N+1, self.W_down_generator_size)
        self.neighbor_sampling_size = args.neighbor_sampling_size
        
        self.lstm = nn.LSTM(self.hidden_units, self.hidden_units, self.num_G_layer)
        self.init_lin_1 = nn.Linear(self.noise_dim, self.hidden_units)
        self.init_lin_2_h = nn.Linear(self.hidden_units, self.hidden_units)
        self.init_lin_2_c = nn.Linear(self.hidden_units, self.hidden_units)
        self.lin_node_type = nn.Linear(self.hidden_units, self.node_classes)

    def reverse_sampling(self, dist):
        dist_weight = torch.exp(-dist)
        return torch.multinomial(dist_weight, 1)[0]
        
    def forward(self, z):
        outputs_type, outputs_node, init_c, init_h = [], [], [], []
        for _ in range(self.num_G_layer):
            intermediate = torch.tanh(self.init_lin_1(z))
            init_c.append(torch.tanh(self.init_lin_2_c(intermediate)))
            init_h.append(torch.tanh(self.init_lin_2_h(intermediate)))
        # Initialize an input tensor
        inputs = Variable(torch.zeros((self.batch_size, self.hidden_units))).to(self.device)
        hidden = (torch.stack(init_c, dim=0), torch.stack(init_h, dim=0))

        # LSTM time steps
        for i in range(self.max_path_len):
            out, hidden = self.lstm(inputs.unsqueeze(0), hidden)
            output_bef = self.lin_node_type(out.squeeze(0))
            output_type = F.gumbel_softmax(output_bef, dim=1, tau=3, hard=True)
            temp_node = []
            for j, x in enumerate(torch.argmax(output_type, dim=1)):
                if x==0:
                    temp_output_node = self.lin_node_0(out.squeeze(0)[j])
                    dist = torch.norm(self.node_embs[x] - temp_output_node, dim=1, p=None)
                    dist = dist.topk(int(dist.shape[0]//1), largest=False)
                    #dist = dist.topk(self.neighbor_sampling_size, largest=False)
                    candidate = dist.indices[self.reverse_sampling(dist.values)]
                    temp_node.append(self.type_0[candidate])
                elif x==1:
                    temp_output_node = self.lin_node_1(out.squeeze(0)[j])
                    dist = torch.norm(self.node_embs[x] - temp_output_node, dim=1, p=None)
                    dist = dist.topk(int(dist.shape[0]//1), largest=False)
                    #dist = dist.topk(self.neighbor_sampling_size, largest=False)
                    candidate = dist.indices[self.reverse_sampling(dist.values)]
                    temp_node.append(self.type_1[candidate])
                elif x==2:
                    temp_output_node = self.lin_node_2(out.squeeze(0)[j])
                    dist = torch.norm(self.node_embs[x] - temp_output_node, dim=1, p=None)
                    dist = dist.topk(int(dist.shape[0]//1), largest=False)
                    #dist = dist.topk(self.neighbor_sampling_size, largest=False)
                    candidate = dist.indices[self.reverse_sampling(dist.values)]
                    temp_node.append(self.type_2[candidate])
                elif x==3:
                    temp_node.append(torch.tensor(self.N).to(self.device))
            temp_node = torch.stack(temp_node)
            output_node = F.one_hot(temp_node.to(int), self.N+1).float()
            outputs_type.append(output_type)
            outputs_node.append(output_node)

            inputs = self.W_down_generator_type(output_type) + self.W_down_generator_node(output_node)

        outputs_type = torch.stack(outputs_type, dim=1)
        outputs_node = torch.stack(outputs_node, dim=1)

        return outputs_type, outputs_node


class Discriminator(nn.Module):
    def __init__(self, N, device, args):
        super(Discriminator, self).__init__()
        self.N = N
        self.device = device
        self.batch_size = args.batch_size
        self.hidden_units = args.hidden_units
        self.W_down_discriminator_size = args.W_down_discriminator_size
        self.max_path_len = args.max_path_len
        if args.dataset in ["syn_100", "syn_200", "syn_500"]:
            self.node_classes = 4
        else:
            self.node_classes = 5
        
        self.lin_in_type = nn.Linear(self.max_path_len*self.node_classes, self.hidden_units)
        self.lstm_type = nn.Linear(self.W_down_discriminator_size, self.hidden_units, 4)
        self.lin_out_3_type = nn.Linear(self.hidden_units, 1)

        self.lin_in_node = nn.Linear(self.max_path_len*(self.N+1), self.hidden_units)
        self.lstm_node = nn.Linear(self.hidden_units, self.hidden_units, 4)
        self.lin_out_3_node = nn.Linear(self.hidden_units, 1)

    def forward(self, z):
        input_type, input_node = z[0], z[1]
        # Discriminator score of node type
        input_type = torch.tanh(self.lin_in_type(input_type.view(self.batch_size, -1)))
        output_type = torch.tanh(self.lstm_type(input_type))
        output_type = self.lin_out_3_type(output_type)
        # Discriminator score of node sequence
        input_node = torch.tanh(self.lin_in_node(input_node.view(self.batch_size, -1)))
        output_node = torch.tanh(self.lstm_node(input_node))
        output_node = self.lin_out_3_node(output_node)
        
        return output_type + output_node


def generate_walks():
    print("Start Generating Graph...")
    transitions_per_walk = 4-1
    transitions_per_iter = 20e4
    eval_transitions = 80e7
    sample_many_count = int(np.round(transitions_per_iter/transitions_per_walk))
    n_eval_walks = eval_transitions/transitions_per_walk
    n_eval_iters = int(np.round(n_eval_walks/sample_many_count))

    smpls_type, smpls_node = [], []
    for i in range(n_eval_iters):
        initial_noise = utils.make_noise((args.batch_size, args.noise_dim), args.noise_type).to(device)
        synthetic_type, synthetic_node = generator(initial_noise)
        synthetic_type = torch.argmax(synthetic_type.cpu(), dim=2).numpy().astype(np.int32)
        synthetic_node = torch.argmax(synthetic_node.cpu(), dim=2).numpy().astype(np.int32)
        smpls_type += utils.delete_from_tail(synthetic_type, 3)
        smpls_node += utils.delete_from_tail(synthetic_node, _N)
        if i % 100 == 0:
            print("Done, generating {} of {} batches meta-paths...".format(i, n_eval_iters))

    return smpls_type, smpls_node


def frequent_meta_path_pattern(smpls_type):
    bb = []
    for i in smpls_type:
        bb.append(tuple(i))
    count = Counter(bb)

    return count.most_common()
    

def meta_path_frequency(smpls_type_2, smpls_type_3, smpls_type_4):
    len_2 = frequent_meta_path_pattern(smpls_type_2)
    len_3 = frequent_meta_path_pattern(smpls_type_3)
    len_4 = frequent_meta_path_pattern(smpls_type_4)
    len_2, len_3, len_4 = len_2[:2*len(len_2)//3], len_3[:2*len(len_3)//3], len_4[:2*len(len_4)//3]
    
    return dict(len_2 + len_3 + len_4)


def evaluation(orig_A, test_A, syn_A):
    orig_G = nx.from_scipy_sparse_matrix(test_A)
    syn_G = nx.from_numpy_matrix(syn_A)
    print("========== Real Graph ==========")
    print(nx.info(orig_G))
    print("========== Generated Graph ==========")
    print(nx.info(syn_G))
    print("")
    print("Clustering coefficient ratio for real graph: ", nx.average_clustering(orig_G))
    print("Clustering coefficient ratio for generated graph: ", nx.average_clustering(syn_G))

    print("Triangle count for real graph: ", utils.statistics_triangle_count(orig_G))
    print("Triangle count for generated graph: ", utils.statistics_triangle_count(syn_G))
    
    print("LCC for real graph: ", len(utils.statistics_LCC(test_A)))
    print("LCC for generated graph: ", len(utils.statistics_LCC(syn_A)))
    
    orig_G_edge = set(orig_G.edges())
    syn_G_edge = set(syn_G.edges())
    intersecting_edges = orig_G_edge & syn_G_edge
    print("Edge Overlap Rate: ", '{:.2%}'.format(len(intersecting_edges)/len(orig_G_edge)))
    
    print("Statistical Powerlaw Distribution Coeff. for real Graph： ", utils.statistics_power_law_alpha(test_A.toarray()))
    print("Statistical Powerlaw Distribution Coeff. for generated Graph： ", utils.statistics_power_law_alpha(syn_A))
    
    # Node Degree Distribution
    test_degree_sequence = sorted([d for n, d in orig_G.degree()], reverse=True)  # degree sequence
    syn_degree_sequence = sorted([d for n, d in syn_G.degree()], reverse=True)  # degree sequence
    mmd_test = MMDStatistic(len(test_degree_sequence), len(syn_degree_sequence))
    test_degree_sequence = torch.tensor(test_degree_sequence).unsqueeze(-1)
    syn_degree_sequence = torch.tensor(syn_degree_sequence).unsqueeze(-1)

    print("MMD distance for node degree distribution: {}".format(mmd_test(
        test_degree_sequence, syn_degree_sequence, alphas=[4.], ret_matrix=False)))
    print("Degree Assortativity \t\t real graph: {}".format(
        nx.degree_assortativity_coefficient(orig_G)))
    print("Degree Assortativity \t\t generated graph: {}".format(
        nx.degree_assortativity_coefficient(syn_G)))
    
    return (nx.average_clustering(syn_G), utils.statistics_triangle_count(syn_G), utils.statistics_LCC(syn_A), utils.statistics_power_law_alpha(syn_A), mmd_test(test_degree_sequence, syn_degree_sequence, alphas=[4.], ret_matrix=False), nx.degree_assortativity_coefficient(syn_G))

print("Loading adj matrix and features...")
_A_obs, _X_obs, _N, lcc, node_type, test_graph = data_preprocess(val_share, test_share, seed)
real_walks = HINDataset(_X_obs, _N, args.max_path_len, args.dataset)
dataloader = DataLoader(real_walks, shuffle=True, batch_size=args.batch_size, num_workers=0, drop_last=True)

print("Reading Node Embeddings...")
node_embs = read_node_embeddings(node_type, lcc, device)

print("Initialize Generator and Discriminator")
generator = Generator(node_type, _N, node_embs, device, args).to(device)
discriminator = Discriminator(_N, device, args).to(device)

optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=args.lr_gen)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=args.lr_dis)


def train():
    batches_done = 0
    print("Start Training...")
    for epoch in range(args.n_epochs):
        for i, (real_node_type, real_node_seq) in enumerate(dataloader):
            # Configure input
            real_node_type = Variable(real_node_type.float()).to(device)
            real_node_seq = Variable(real_node_seq.float()).to(device)
            initial_noise = utils.make_noise((args.batch_size, args.noise_dim), args.noise_type).to(device)
            # ---------------------
            #  Train Discriminator
            # ---------------------
            discriminator.train()
            optimizer_D.zero_grad()
            fake_result = generator(initial_noise)
            fake_type = fake_result[0].detach()
            fake_node = fake_result[1].detach()
            loss_D = torch.mean(discriminator((fake_type, fake_node))) -\
                torch.mean(discriminator((real_node_type, real_node_seq)))

            loss_D.backward()
            optimizer_D.step()

            # Clip weights of discriminator
            for p in discriminator.parameters():
                p.data.clamp_(-args.clip_value, args.clip_value)
                
            # Train the generator every n_critic iterations
            if i % args.n_critic == 0:
                # -----------------
                #  Train Generator
                # -----------------
                optimizer_G.zero_grad()

                # Generate a batch of random walks
                syn_types, syn_node_seq = generator(initial_noise)
                # Adversarial loss
                loss_G = -torch.mean(discriminator((syn_types, syn_node_seq)))
                # Loss back-propagation
                loss_G.backward()
                optimizer_G.step()
            
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch+1, args.n_epochs, batches_done % len(dataloader), len(dataloader), loss_D.item(), loss_G.item()))
            batches_done += 1


def write_to_gephi(test_A, syn_A, dataset, node_attribute):
    orig_G = nx.from_scipy_sparse_matrix(test_A)
    syn_G = nx.from_numpy_matrix(syn_A)
    for i in orig_G.nodes:
        orig_G.nodes[i]['class'] = node_attribute[i]
        syn_G.nodes[i]['class'] = node_attribute[i]
        
    nx.write_gexf(orig_G, "results/{}_real.gexf".format(dataset))
    nx.write_gexf(syn_G, "results/{}_hgen.gexf".format(dataset))
    

def getting_metapaths(adj_matrix):
    try:
        aa = adj_matrix.toarray()
    except:
        aa = adj_matrix
    edge_dir = {}
    for i, x in enumerate(aa):
        edge_dir[i] = list(np.nonzero(x)[0])
    
    meta_path_length_2, meta_path_length_3, meta_path_length_4 = [], [], []
    for i in range(len(edge_dir)):
        for p in utils.findAllPaths(i, lambda n: edge_dir[n], depth=2):
            meta_path_length_2.append(p)
        for p in utils.findAllPaths(i, lambda n: edge_dir[n], depth=3):
            meta_path_length_3.append(p)
        for p in utils.findAllPaths(i, lambda n: edge_dir[n], depth=4):
            meta_path_length_4.append(p)
        if i % 1000 == 0:
            print("Done, {} of {}".format(i, len(edge_dir)))
    
    return meta_path_length_2, meta_path_length_3, meta_path_length_4


def heterogeneous_statistics(test_graph, syn_graph):
    print("Getting real meta-paths...")
    real_meta_paths = getting_metapaths(test_graph)
    print("Getting generated meta-paths...")
    generated_meta_paths = getting_metapaths(syn_graph)
    

if __name__ == '__main__':
    # Initialize the time
    starting_time = time.time()
    timestr = time.strftime("%Y%m%d-%H%M%S")
    print("The training has been started at {}".format(starting_time))
    train()
    if args.save:
        torch.save(generator.state_dict(), "model/{}_{}".format(int(time.time()), args.dataset))

    smpls_type, smpls_node = generate_walks()
    #smpls_type, smpls_node = pickle.load(open("results/smpls_type_dblp.p","rb")), pickle.load(open("results/smpls_node_dblp.p","rb"))
    smpls_type_2 = [i for i in smpls_type if i.shape[0] == 2 and 3 not in i]
    smpls_type_3 = [i for i in smpls_type if i.shape[0] == 3 and 3 not in i]
    smpls_type_4 = [i for i in smpls_type if i.shape[0] == 4 and 3 not in i]
    
    smpls_node_2 = [i for i in smpls_node if i.shape[0] == 2 and _N not in i]
    smpls_node_3 = [i for i in smpls_node if i.shape[0] == 3 and _N not in i]
    smpls_node_4 = [i for i in smpls_node if i.shape[0] == 4 and _N not in i]
    
    meta_path_freq = meta_path_frequency(smpls_type_2, smpls_type_3, smpls_type_4)
    
    score_matrix = utils.score_matrix_from_random_walks(smpls_node_2, _N)
    score_matrix += utils.score_matrix_from_random_walks(smpls_node_3, _N)
    score_matrix += utils.score_matrix_from_random_walks(smpls_node_4, _N)
    score_matrix = score_matrix.tocsr()
    syn_graph = utils.heterogeneous_graph_assemble(score_matrix, test_graph.sum(), meta_path_freq, node_type)
    
    t = time.time() - starting_time
    print('Took {} seconds so far...'.format(int(t)))
    
    evaluation(_A_obs, test_graph, syn_graph)
    
    uniqueness = []
    for k in range(5):
        generated_graphs = []
        for _ in range(10):
            generated_graphs.append(nx.from_numpy_matrix(
                utils.heterogeneous_graph_assemble(score_matrix, test_graph.sum(), meta_path_freq, node_type)))
    
        generated_graph_edges = [list(i.edges()) for i in generated_graphs]
        uniqueness_temp = []
        for i, x in enumerate(generated_graph_edges[1:]):
            overlaps = set(generated_graph_edges[0]).intersection(set(x))
            uniqueness_temp.append(len(overlaps)/len(generated_graph_edges[0]))
        uniqueness.append(1 - np.mean(uniqueness_temp))
    
    
    print("Getting real meta-paths...")
    real_meta_paths = getting_metapaths(test_graph)
    print("Getting generated meta-paths...")
    generated_meta_paths = getting_metapaths(syn_graph)

    metapath_len_2 = [[tuple(sorted([_X_obs[j] for j in i])) for i in real_meta_paths[0]], [tuple(sorted([_X_obs[j] for j in i])) for i in generated_meta_paths[0]]]
    metapath_len_3 = [[tuple(sorted([_X_obs[j] for j in i])) for i in real_meta_paths[1]], [tuple(sorted([_X_obs[j] for j in i])) for i in generated_meta_paths[1]]]
    metapath_len_4 = [[tuple(sorted([_X_obs[j] for j in i])) for i in real_meta_paths[2]], [tuple(sorted([_X_obs[j] for j in i])) for i in generated_meta_paths[2]]]
    
