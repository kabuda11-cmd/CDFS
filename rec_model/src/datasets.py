import copy
import torch
from utils import neg_sample
from torch.utils.data import Dataset
import random

class SASRecDataset(Dataset):

    def __init__(self, args, user_seq, test_neg_items=None, mode='train'):
        self.args = args
        self.user_seq = user_seq
        self.test_neg_items = test_neg_items
        self.mode = mode
        self.max_len = args.max_seq_length

    def _data_sample_rec_task(self, user_id, items, input_ids, target_pos, answer):
        target_neg = []
        seq_set = set(items)
        for _ in input_ids:
            target_neg.append(neg_sample(seq_set, self.args.item_size))
        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        input_ids = input_ids[-self.max_len:]
        target_pos = target_pos[-self.max_len:]
        target_neg = target_neg[-self.max_len:]

        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len

        if self.test_neg_items is not None:
            test_samples = self.test_neg_items

            cur_rec_tensors = (
                torch.tensor(user_id, dtype=torch.long), # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(test_samples, dtype=torch.long),
            )
        else:
            cur_rec_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
            )

        return cur_rec_tensors

    def __getitem__(self, index):

        user_id = index
        items = self.user_seq[index]

        if self.mode == "train":
            #aug_input_ids,aug_target_pos,_ = self.generate_subsequences(items[:-2],5)
            input_ids = items[:-3]
            target_pos = items[1:-2]
            answer = [0]
            # if input_ids not in aug_input_ids:
            #     aug_input_ids.append(input_ids)
            #     aug_target_pos.append(target_pos)
            # train_data=[]
            # for input_ids, target_ids in zip(aug_input_ids, aug_target_pos):
            #     train_data.append(self._data_sample_rec_task(user_id, items, input_ids, \
            #                                       target_ids, answer))
            # while len(train_data) < 6:
            #     if train_data:  # 确保 train_data 不为空
            #         train_data.append(train_data[-1])  # 复制最后一个元素
            # return  train_data

        elif self.mode == 'valid':
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]

        else:
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [items[-1]]


        return self._data_sample_rec_task(user_id, items, input_ids, \
                                            target_pos, answer)

    def __len__(self):
        return len(self.user_seq)

    def generate_subsequences(self,item_ids, n):
        sub_sequences = []
        next_steps = []
        last_items = []

        for _ in range(n):
            start = random.randint(0, len(item_ids) - 3)
            end = random.randint(start + 1, len(item_ids) - 2)
            sub_seq = item_ids[start:end + 1]
            next_seq = item_ids[start + 1:end + 2]
            if sub_seq not in sub_sequences:
                sub_sequences.append(sub_seq)
                next_steps.append(next_seq)
                last_items.append(next_seq[-1])

        return sub_sequences, next_steps, last_items