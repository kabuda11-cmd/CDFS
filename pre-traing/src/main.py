# -*- coding: utf-8 -*-

import os
import numpy as np
import random
import torch
import time
import argparse

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from datasets import RecWithContrastiveLearningDataset

from trainers import CDFSTrainer
from models import SASRecModel, OfflineItemSimilarity, OnlineItemSimilarity, generate_item_embeddings,load_weight,generate_SeqtoSeq
from utils import EarlyStopping, get_user_seqs, get_item2attribute_json, check_path, set_seed
import pydevd_pycharm
#pydevd_pycharm.settrace('172.21.176.1', port=65440, stdoutToServer=True, stderrToServer=True)

def show_args_info(args):
    print(f"--------------------Configure Info:------------")
    for arg in vars(args):
        print(f"{arg:<30} : {getattr(args, arg):>35}")

def main():
    parser = argparse.ArgumentParser()
    #system args
    parser.add_argument('--data_dir', default='../data/', type=str)
    parser.add_argument('--output_dir', default='output_cl/', type=str)
    parser.add_argument('--data_name', default='Beauty', type=str)
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--model_idx', default=0, type=int, help="model idenfier 10, 20, 30...")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")

    #data augmentation args
    parser.add_argument('--noise_ratio', default=0.0, type=float, \
                        help="percentage of negative interactions in a sequence - robustness analysis")
    parser.add_argument('--training_data_ratio', default=1.0, type=float, \
                        help="percentage of training samples used for training - robustness analysis")
    parser.add_argument('--augment_threshold', default=4, type=int, \
                        help="control augmentations on short and long sequences.\
                        default:-1, means all augmentations types are allowed for all sequences.\
                        For sequence length < augment_threshold: Insert, and Substitute methods are allowed \
                        For sequence length > augment_threshold: Crop, Reorder, Substitute, and Mask \
                        are allowed.")
    parser.add_argument('--similarity_model_name', default='ItemCF_IUF', type=str, \
                        help="Method to generate item similarity score. choices: \
                        Random, ItemCF, ItemCF_IUF(Inverse user frequency), Item2Vec, LightGCN")
    # parser.add_argument("--augmentation_warm_up_epoches", type=float, default=160, \
    #                     help="number of epochs to switch from \
    #                     memory-based similarity model to \
    #                     hybrid similarity model.")
    parser.add_argument('--base_augment_type', default='random', type=str, \
                        help="default data augmentation types. Chosen from: \
                        mask, crop, reorder, substitute, insert, random, \
                        substituteEPS and insertEPS (for multi-view).")
    parser.add_argument('--augment_type_for_short', default='SIM', type=str, \
                        help="data augmentation types for short sequences. Chosen from: \
                        SI, SIM, SIR, SIC, SIMR, SIMC, SIRC, SIMRC.")
    parser.add_argument("--tao", type=float, default=0.2, help="crop ratio for crop operator")
    parser.add_argument("--gamma", type=float, default=0.7, help="mask ratio for mask operator")
    parser.add_argument("--beta", type=float, default=0.2, help="reorder ratio for reorder operator") 
    parser.add_argument("--substitute_rate", type=float, default=0.1, \
                        help="substitute ratio for substitute operator")
    parser.add_argument("--insert_rate", type=float, default=0.4, \
                        help="insert ratio for insert operator")
    parser.add_argument("--max_insert_num_per_pos", type=int, default=1, \
                        help="maximum insert items per position for insert operator - not studied")

    parser.add_argument('--sim_path', type=str,default='123',
                        help='file of sim_path.')
    ## contrastive learning task args
    parser.add_argument('--temperature', default= 1.0, type=float,
                        help='softmax temperature (default:  1.0) - not studied.')
    parser.add_argument('--n_views', default=2, type=int, metavar='N',
                        help='Number of augmented data for each sequence - not studied.')

    # model args
    parser.add_argument("--model_name", default='CDFS', type=str)
    parser.add_argument("--hidden_size", type=int, default=64, help="hidden size of transformer model")
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
    parser.add_argument('--num_attention_heads', default=2, type=int)
    parser.add_argument('--hidden_act', default="gelu", type=str) # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument('--max_seq_length', default=50, type=int)

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=256, help="number of batch_size")
    # parser.add_argument("--epochs", type=int, default=300, help="number of epochs")
    parser.add_argument("--cl_epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=1, type=int)
    # parser.add_argument("--cf_weight", type=float, default=1.0, \
    #                     help="weight of contrastive learning task")
    # parser.add_argument("--rec_weight", type=float, default=0, \
    #                     help="weight of contrastive learning task")

    #learning related
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")

    # parser.add_argument('--gen_h', action='store_true')
    parser.add_argument('--seq_path', type=str, default=' ')
    #generate res
    args = parser.parse_args()

    set_seed(args.seed)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    print("Using Cuda:", torch.cuda.is_available())
    args.data_file = args.data_dir + args.data_name + '.txt'

    # user_seq, max_item, valid_rating_matrix, test_rating_matrix = \
    #     get_user_seqs(args.data_file)
    user_seq, max_item, _, _ = \
        get_user_seqs(args.data_file)
    args.item_size = max_item + 2
    args.mask_id = max_item + 1
    args.sim_path = './sim_dir/'+ args.data_name + '.json'
    # save model args
    args_str = f'{args.model_name}-{args.data_name}-{args.model_idx}'
    args.log_file = os.path.join(args.output_dir, args_str + '.txt')
    
    show_args_info(args)

    with open(args.log_file, 'a') as f:
        f.write(str(args) + '\n')

    checkpoint = args_str + '.pt'
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)
    # -----------   pre-computation for item similarity   ------------ #
    args.similarity_model_path = os.path.join(args.data_dir,\
                            args.data_name+'_'+args.similarity_model_name+'_similarity.pkl')

    offline_similarity_model = OfflineItemSimilarity(data_file=args.data_file,
                            similarity_path=args.similarity_model_path,
                            model_name=args.similarity_model_name,
                            dataset_name=args.data_name)
    args.offline_similarity_model = offline_similarity_model

    # -----------   online based on shared item embedding for item similarity --------- #

    # training data for node classification
    train_dataset = RecWithContrastiveLearningDataset(args, 
                                    user_seq[:int(len(user_seq)*args.training_data_ratio)], \
                                    data_type='train')
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    model = SASRecModel(args=args)

    trainer = CDFSTrainer(model, train_dataloader, args)
    args.seq_path=args.data_dir + args.data_name+'_seq.json'
    print(f'Pretrain CDFS')
    for epoch in range(args.cl_epochs):
        trainer.train(epoch)
    torch.save(trainer.projection.state_dict(), args.pro_checkpoint_path)
    print('---------------Change to test_rating_matrix!-------------------')
    print(args_str)
    print(result_info)
    with open(args.log_file, 'a') as f:
        f.write(args_str + '\n')
        f.write(result_info + '\n')
main()
