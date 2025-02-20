# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm
import random

import torch
import torch.nn as nn
from torch.optim import Adam

from torch.utils.data import DataLoader, RandomSampler
from datasets import RecWithContrastiveLearningDataset
from modules import NCELoss, NTXent
from utils import recall_at_k, ndcg_k, get_metric, get_user_seqs, nCr

class Trainer:
    def __init__(self, model, train_dataloader,
                 args):

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model

        self.total_augmentaion_pairs = nCr(self.args.n_views, 2)
        #projection head for contrastive learn task
        self.projection = nn.Sequential(nn.Linear(self.args.max_seq_length*self.args.hidden_size, \
                                        512, bias=False), nn.BatchNorm1d(512), nn.ReLU(inplace=True), 
                                        nn.Linear(512, self.args.hidden_size*2, bias=True))
        if self.cuda_condition:
            self.model.cuda()
            self.projection.cuda()
        # Setting the train and test data loader
        self.train_dataloader = train_dataloader

        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

        self.cf_criterion = NCELoss(self.args.temperature, self.device)
        # self.cf_criterion = NTXent()
        print("self.cf_criterion:", self.cf_criterion.__class__.__name__)

        
    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader)

    def iteration(self, epoch, dataloader, full_sort=False, train=True):
        raise NotImplementedError


    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)


class CDFSTrainer(Trainer):

    def __init__(self, model,
                 train_dataloader,
                 args):
        super(CDFSTrainer, self).__init__(
            model,
            train_dataloader,
            args
        )

    def _one_pair_contrastive_learning(self, inputs):
        '''
        contrastive learning given one pair sequences (batch)
        inputs: [batch1_augmented_data, batch2_augmentated_data]
        '''
        cl_batch = torch.cat(inputs, dim=0)
        cl_batch = cl_batch.to(self.device)
        cl_sequence_output = self.model.transformer_encoder(cl_batch)
        cl_sequence_flatten = cl_sequence_output.view(cl_batch.shape[0], -1)
        cf_output = self.projection(cl_sequence_flatten)
        batch_size = cl_batch.shape[0]//2
        cl_output_slice = torch.split(cf_output, batch_size)
        cl_loss = self.cf_criterion(cl_output_slice[0], 
                                cl_output_slice[1])
        return cl_loss

    def iteration(self, epoch, dataloader, full_sort=True, train=True):

        str_code = "train" if train else "test"

        # Setting the tqdm progress bar

        if train:
            self.model.train()
            rec_avg_loss = 0.0
            cl_individual_avg_losses = [0.0 for i in range(self.total_augmentaion_pairs)]
            cl_sum_avg_loss = 0.0
            joint_avg_loss = 0.0

            print(f"rec dataset length: {len(dataloader)}")
            rec_cf_data_iter = tqdm(enumerate(dataloader), total=len(dataloader))

            for i, (rec_batch, cl_batches) in rec_cf_data_iter:
                '''
                rec_batch shape: key_name x batch_size x feature_dim
                cl_batches shape: 
                    list of n_views x batch_size x feature_dim tensors
                '''
                # 0. batch_data will be sent into the device(GPU or CPU)
                rec_batch = tuple(t.to(self.device) for t in rec_batch)
                _, input_ids, target_pos, target_neg, _ = rec_batch

                # # ---------- recommendation task ---------------#
                # sequence_output = self.model.transformer_encoder(input_ids)
                # ---------- contrastive learning task -------------#
                cl_losses = []
                for cl_batch in cl_batches:
                    cl_loss = self._one_pair_contrastive_learning(cl_batch)
                    cl_losses.append(cl_loss)

                joint_loss = 0
                for cl_loss in cl_losses:
                    joint_loss += cl_loss
                self.optim.zero_grad()
                joint_loss.backward()
                self.optim.step()

                for i, cl_loss in enumerate(cl_losses):
                    cl_individual_avg_losses[i] += cl_loss.item()
                    cl_sum_avg_loss += cl_loss.item()
                joint_avg_loss += joint_loss.item()


            post_fix = {
                "epoch": epoch,
                "cl_avg_loss": '{:.4f}'.format(cl_sum_avg_loss / (len(rec_cf_data_iter)*self.total_augmentaion_pairs)),
            }
            for i, cl_individual_avg_loss in enumerate(cl_individual_avg_losses):
                post_fix['cl_pair_'+str(i)+'_loss'] = '{:.4f}'.format(cl_individual_avg_loss / len(rec_cf_data_iter))

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')

