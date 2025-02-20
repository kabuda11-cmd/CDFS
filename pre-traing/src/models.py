# -*- coding: utf-8 -*-
import  math
import os
import pickle
from tqdm import tqdm
import random
import copy

import numpy as np
import json
import torch
import torch.nn as nn
import gensim
from utils import MLP,load_npy
from modules import Encoder, LayerNorm

class SASRecModel(nn.Module):
    def __init__(self, args):
        super(SASRecModel, self).__init__()
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.item_encoder = Encoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args

        self.criterion = nn.BCELoss(reduction='none')
        self.apply(self.init_weights)


    # Positional Embedding
    def add_position_embedding(self, sequence):

        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    # model same as SASRec
    def transformer_encoder(self, input_ids):

        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1) # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        sequence_emb = self.add_position_embedding(input_ids)

        item_encoded_layers = self.item_encoder(sequence_emb,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True)

        sequence_output = item_encoded_layers[-1]
        return sequence_output

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

def load_weight(model, file_name):
    model.load_state_dict(torch.load(file_name))

def generate_item_embeddings(model, args):

    json_file_path = args.json_file_path
    output_npy_path = args.output_npy_path
    max_co = args.max_co

    with open(json_file_path, 'r') as f:
        item_data = json.load(f)
    all_item_embeddings = []
    for item_id, co_occurring_ids in item_data.items():
        if len(co_occurring_ids) > max_co:
            co_occurring_ids = co_occurring_ids[:max_co]
        else:
            co_occurring_ids = [0] * (max_co - len(co_occurring_ids)) + co_occurring_ids

        input_tensor = torch.tensor(co_occurring_ids, dtype=torch.long).unsqueeze(0)  # 添加 batch 维度
        input_tensor = input_tensor.to('cuda')  # 假设你使用 GPU

        with torch.no_grad():  # 禁用梯度计算，节省内存
            item_embedding = model.transformer_encoder(input_tensor)
        n = len(co_occurring_ids)  # 当前输入的长度
        selected_embeddings = item_embedding[:, -n:, :].cpu().numpy()  # 选择最后 n 个 token 的 embedding
        all_item_embeddings.append(selected_embeddings.flatten())  # 扁平化后加入列表

    all_item_embeddings = np.array(all_item_embeddings)
    np.save(output_npy_path, all_item_embeddings)

def generate_SeqtoSeq(mlp,model, user_seq, args):
    result_dict = {}
    device = next(model.parameters()).device
    user_seq_tensor = [torch.tensor(seq[:-3], dtype=torch.long).unsqueeze(0).to(device) for seq in user_seq]
    all_representations = []
    mlp.eval()
    for seq in user_seq_tensor:
        # 如果 seq 的长度小于 args.max_length，则在前面补 0
        if seq.size(1) < args.max_seq_length:
            padding = torch.zeros((1, args.max_seq_length - seq.size(1)), dtype=torch.long).to(seq.device)
            seq = torch.cat([padding, seq], dim=1)
        # 如果 seq 的长度大于 args.max_length，则从左边截断到 args.max_length
        elif seq.size(1) > args.max_seq_length:
            seq = seq[:, -args.max_seq_length:]

        with torch.no_grad():
            seq_repr = model.transformer_encoder(seq)  # 假设 transformer_encoder 是模型的方法
            seq_repr = torch.flatten(seq_repr)
            seq_repr = mlp(seq_repr.unsqueeze(0))
            all_representations.append(seq_repr.squeeze(0).cpu().numpy())

    all_representations = np.array(all_representations)
    norm_representations = all_representations / np.linalg.norm(all_representations, axis=1, keepdims=True)
    similarity_matrix = np.matmul(norm_representations, norm_representations.T)

    for i, seq in enumerate(user_seq_tensor):
        similarity_vector = similarity_matrix[i]
        top_indices = np.argsort(-similarity_vector)[1:6]
        top_sequences = [user_seq[idx] for idx in top_indices]
        result_dict[str(seq)] = {
            "representation": all_representations[i].tolist(),
            "most_similar_sequences": top_sequences
        }

    with open(args.seq_path, 'w') as f:
        json.dump(result_dict, f, indent=4)

class OnlineItemSimilarity:

    def __init__(self, item_size):
        self.item_size = item_size
        self.item_embeddings = None
        self.cuda_condition = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")
        self.total_item_list = torch.tensor([i for i in range(self.item_size)],
                                            dtype=torch.long).to(self.device)
        # self.max_score, self.min_score = self.get_maximum_minimum_sim_scores()
        
    def update_embedding_matrix(self, item_embeddings):
        print(item_embeddings)
        self.item_embeddings = copy.deepcopy(item_embeddings)
        self.base_embedding_matrix =self.item_embeddings(self.total_item_list)
        self.max_score, self.min_score = self.get_maximum_minimum_sim_scores()

    def get_maximum_minimum_sim_scores(self):
        max_score, min_score = -1, 100
        for item_idx in range(1, self.item_size):
            try:
                item_vector = self.item_embeddings(torch.tensor(item_idx).to(self.device)).view(-1, 1)
                item_similarity = torch.mm(self.base_embedding_matrix, item_vector).view(-1)
                max_score = max(torch.max(item_similarity), max_score)
                min_score = min(torch.min(item_similarity), min_score)
            except:
                print("ssssss")
                continue
        return max_score, min_score
        
    def most_similar(self, item_idx, top_k=1, with_score=False):
        

        item_idx = torch.tensor(item_idx, dtype=torch.long).to(self.device)
        item_vector = self.item_embeddings(item_idx).view(-1, 1)
        item_similarity = torch.mm(self.base_embedding_matrix, item_vector).view(-1)
        item_similarity = (item_similarity - self.min_score) / (self.max_score - self.min_score)
        #remove item idx itself
        values, indices = item_similarity.topk(top_k+1)
        if with_score:
            item_list = indices.tolist()
            score_list = values.tolist()
            if item_idx in item_list:
                idd = item_list.index(item_idx)
                item_list.remove(item_idx)
                score_list.pop(idd)
            return list(zip(item_list, score_list))
        item_list = indices.tolist()
        if item_idx in item_list:
            item_list.remove(item_idx)
        return item_list

class OfflineItemSimilarity:
    def __init__(self, data_file=None, similarity_path=None, model_name='ItemCF', \
        dataset_name='Sports_and_Outdoors'):
        self.dataset_name = dataset_name
        self.similarity_path = similarity_path
        # train_data_list used for item2vec, train_data_dict used for itemCF and itemCF-IUF
        self.train_data_list, self.train_item_list, self.train_data_dict = self._load_train_data(data_file)
        self.model_name = model_name
        self.similarity_model = self.load_similarity_model(self.similarity_path)
        self.max_score, self.min_score = self.get_maximum_minimum_sim_scores()
        
    def get_maximum_minimum_sim_scores(self):
        max_score, min_score = -1, 100
        for item in self.similarity_model.keys():
            for neig in self.similarity_model[item]:
                sim_score = self.similarity_model[item][neig]
                max_score = max(max_score, sim_score)
                min_score = min(min_score, sim_score)
        return max_score, min_score
    
    def _convert_data_to_dict(self, data):
        """
        split the data set
        testdata is a test data set
        traindata is a train set
        """
        train_data_dict = {}
        for user,item,record in data:
            train_data_dict.setdefault(user,{})
            train_data_dict[user][item] = record
        return train_data_dict

    def _save_dict(self, dict_data, save_path = './similarity.pkl'):
        print("saving data to ", save_path)
        with open(save_path, 'wb') as write_file:
            pickle.dump(dict_data, write_file)

    def _load_train_data(self, data_file=None):
        """
        read the data from the data file which is a data set
        """
        train_data = []
        train_data_list = []
        train_data_set_list = []
        for line in open(data_file).readlines():
            userid, items = line.strip().split(' ', 1)
            # only use training data
            items = items.split(' ')[:-3]
            train_data_list.append(items)
            train_data_set_list += items
            for itemid in items:
                train_data.append((userid,itemid,int(1)))
        return train_data_list, set(train_data_set_list), self._convert_data_to_dict(train_data)

    def _generate_item_similarity(self,train=None, save_path='./'):
        """
        calculate co-rated users between items
        """
        print("getting item similarity...")
        train = train or self.train_data_dict
        C = dict()
        N = dict()

        if self.model_name in ['ItemCF', 'ItemCF_IUF']:
            print("Step 1: Compute Statistics")
            data_iter = tqdm(enumerate(train.items()), total=len(train.items()))
            for idx, (u, items) in data_iter:
                if self.model_name == 'ItemCF':
                    for i in items.keys():
                        N.setdefault(i,0)
                        N[i] += 1
                        for j in items.keys():
                            if i == j:
                                continue
                            C.setdefault(i,{})
                            C[i].setdefault(j,0)
                            C[i][j] += 1
                elif self.model_name == 'ItemCF_IUF':
                    for i in items.keys():
                        N.setdefault(i,0)
                        N[i] += 1
                        for j in items.keys():
                            if i == j:
                                continue
                            C.setdefault(i,{})
                            C[i].setdefault(j,0)
                            C[i][j] += 1 / math.log(1 + len(items) * 1.0)
            self.itemSimBest = dict()
            print("Step 2: Compute co-rate matrix")
            c_iter = tqdm(enumerate(C.items()), total=len(C.items()))
            for idx, (cur_item, related_items) in c_iter:
                self.itemSimBest.setdefault(cur_item,{})
                for related_item, score in related_items.items():
                    self.itemSimBest[cur_item].setdefault(related_item,0);
                    self.itemSimBest[cur_item][related_item] = score / math.sqrt(N[cur_item] * N[related_item])
            self._save_dict(self.itemSimBest, save_path=save_path)
        elif self.model_name == 'Item2Vec':
            # details here: https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/word2vec.py
            print("Step 1: train item2vec model")
            item2vec_model = gensim.models.Word2Vec(sentences=self.train_data_list,
                                        vector_size=20, window=5, min_count=0, 
                                        epochs=100)
            self.itemSimBest = dict()
            total_item_nums = len(item2vec_model.wv.index_to_key)
            print("Step 2: convert to item similarity dict")
            total_items = tqdm(item2vec_model.wv.index_to_key, total=total_item_nums)
            for cur_item in total_items:
                related_items = item2vec_model.wv.most_similar(positive=[cur_item], topn=20)
                self.itemSimBest.setdefault(cur_item,{})
                for (related_item, score) in related_items:
                    self.itemSimBest[cur_item].setdefault(related_item,0)
                    self.itemSimBest[cur_item][related_item] = score
            print("Item2Vec model saved to: ", save_path)
            self._save_dict(self.itemSimBest, save_path=save_path)
        elif self.model_name == 'LightGCN':
            # train a item embedding from lightGCN model, and then convert to sim dict
            print("generating similarity model..")
            itemSimBest = light_gcn.generate_similarity_from_light_gcn(self.dataset_name)
            print("LightGCN based model saved to: ", save_path)
            self._save_dict(itemSimBest, save_path=save_path)

    def load_similarity_model(self, similarity_model_path):
        if not similarity_model_path:
            raise ValueError('invalid path')
        elif not os.path.exists(similarity_model_path):
            print("the similirity dict not exist, generating...")
            self._generate_item_similarity(save_path=self.similarity_path)
        if self.model_name in ['ItemCF', 'ItemCF_IUF', 'Item2Vec', 'LightGCN']:
            with open(similarity_model_path, 'rb') as read_file:
                similarity_dict = pickle.load(read_file)
            return similarity_dict
        elif self.model_name == 'Random':
            similarity_dict = self.train_item_list
            return similarity_dict

    def most_similar(self, item, top_k=1, with_score=False):
        if self.model_name in ['ItemCF', 'ItemCF_IUF', 'Item2Vec', 'LightGCN']:
            """TODO: handle case that item not in keys"""
            if str(item) in self.similarity_model:
                top_k_items_with_score = sorted(self.similarity_model[str(item)].items(),key=lambda x : x[1], \
                                            reverse=True)[0:top_k]
                if with_score:
                    return list(map(lambda x: (int(x[0]), (float(x[1]) - self.min_score)/(self.max_score -self.min_score)), top_k_items_with_score))
                return list(map(lambda x: int(x[0]), top_k_items_with_score))
            elif int(item) in self.similarity_model:
                top_k_items_with_score = sorted(self.similarity_model[int(item)].items(),key=lambda x : x[1], \
                                            reverse=True)[0:top_k]
                if with_score:
                    return list(map(lambda x: (int(x[0]), (float(x[1]) - self.min_score)/(self.max_score -self.min_score)), top_k_items_with_score))
                return list(map(lambda x: int(x[0]), top_k_items_with_score))
            else:
                item_list = list(self.similarity_model.keys())
                random_items = random.sample(item_list, k=top_k)
                if with_score:
                    return list(map(lambda x: (int(x), 0.0), random_items))
                return list(map(lambda x: int(x), random_items))
        elif self.model_name == 'Random':
            random_items = random.sample(self.similarity_model, k = top_k)
            if with_score:
                return list(map(lambda x: (int(x), 0.0), random_items))
            return list(map(lambda x: int(x), random_items))

if __name__ == '__main__':
    onlineitemsim = OnlineItemSimilarity(item_size=10)
    item_embeddings = nn.Embedding(10, 6, padding_idx=0)
    onlineitemsim.update_embedding_matrix(item_embeddings)
    item_idx = torch.tensor(2, dtype=torch.long)
    similiar_items = onlineitemsim.most_similar(item_idx=item_idx, top_k=1)
    print(similiar_items)