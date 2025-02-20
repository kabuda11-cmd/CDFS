import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, xavier_normal_
# from modules import SelfAttention,NCELoss
from SASRec import SASRecModel
from GRU4Rec import GRU
from utils import load_npy

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish}

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class Attention(nn.Module):
    def __init__(self, args):
        super(Attention, self).__init__()
        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = int(args.hidden_size / args.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, attention_mask):
        q_layer = self.transpose_for_scores(q)
        k_layer = self.transpose_for_scores(k)
        v_layer = self.transpose_for_scores(v)

        attention_scores = torch.matmul(q_layer, k_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, v_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer

class SelfAttention(nn.Module):
    def __init__(self, args):
        super(SelfAttention, self).__init__()
        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = int(args.hidden_size / args.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.q = nn.Linear(args.hidden_size, self.all_head_size)
        self.k = nn.Linear(args.hidden_size, self.all_head_size)
        self.v = nn.Linear(args.hidden_size, self.all_head_size)

        self.attention = Attention(args)
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)

    def forward(self, input_tensor, attention_mask):
        q = self.q(input_tensor)
        k = self.k(input_tensor)
        v = self.v(input_tensor)

        context_layer = self.attention(q, k, v, attention_mask)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class CrossAttention(nn.Module):
    def __init__(self, args):
        super(CrossAttention, self).__init__()
        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = int(args.hidden_size / args.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.q = nn.Linear(args.hidden_size, self.all_head_size)
        self.k = nn.Linear(args.hidden_size, self.all_head_size)
        self.v = nn.Linear(args.hidden_size, self.all_head_size)

        self.attention = Attention(args)
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)

    def forward(self, input_tensor_1, input_tensor_2, attention_mask):
        q = self.q(input_tensor_1)
        k = self.k(input_tensor_2)
        v = self.v(input_tensor_2)

        context_layer = self.attention(q, k, v, attention_mask)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor_1)

        return hidden_states

class Intermediate(nn.Module):
    def __init__(self, args):
        super(Intermediate, self).__init__()
        self.dense_1 = nn.Linear(args.hidden_size, args.hidden_size * 4)
        if isinstance(args.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[args.hidden_act]
        else:
            self.intermediate_act_fn = args.hidden_act

        self.dense_2 = nn.Linear(args.hidden_size * 4, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

    def forward(self, input_tensor):

        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class Project(nn.Module):
    def __init__(self, args):
        super(Project, self).__init__()
        self.LN = LayerNorm(args.hidden_size, eps=1e-12)
        self.dense_1 = nn.Linear(args.hidden_size, args.hidden_size*2)
        self.project_act_fn = ACT2FN[args.hidden_act]
        self.dense_2 = nn.Linear(args.hidden_size*2, args.hidden_size)

    def forward(self, input_tensor):
        hidden_states = self.LN(input_tensor)
        hidden_states = self.dense_1(hidden_states)
        hidden_states = self.project_act_fn(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        return hidden_states

class GRU_Rec(nn.Module):

    def __init__(self, args):
        super(GRU_Rec, self).__init__()
        self.emb_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.gru_layers = nn.GRU(
            input_size=args.hidden_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_hidden_layers,
            bias=False,
            batch_first=True,
        )
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.GRU):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)

    def forward(self, item_seq_emb):
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        gru_output = self.dense(gru_output)
        # the embedding of the predicted item, shape of (batch_size, embedding_size)
        #seq_output = self.gather_indexes(gru_output, item_seq_len - 1)
        return gru_output

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.cross_attention = CrossAttention(args)
        self.ffn_1 = Intermediate(args)
        self.rec_model = args.rec_model
        if args.rec_model == 'GRU4Rec':
            self.self_attention_1 = GRU_Rec(args)
        else:
            self.self_attention_1 = SelfAttention(args)
        self.ffn_2 = Intermediate(args)
        self.layernorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

    def forward(self, seq_emb, cl_emb, position_emb, attention_mask):
        emb = self.cross_attention(seq_emb, cl_emb, attention_mask)
        emb = self.ffn_1(emb)
        hidden_states = self.layernorm(emb+position_emb)
        hidden_states = self.dropout(hidden_states)
        if self.rec_model == 'GRU4Rec':
            hidden_states = self.self_attention_1(hidden_states)
        else:
            hidden_states = self.self_attention_1(hidden_states,attention_mask)
        hidden_states = self.ffn_2(hidden_states)
        return hidden_states

class CDFSModel(nn.Module):
    def __init__(self, args):
        super(CDFSModel, self).__init__()
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.feat_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.model_cl = SASRecModel(args=args)

        self.model_cl.load_state_dict(torch.load(args.model_cl_path))


        for param in self.model_cl.parameters():
            param.requires_grad = False

        #加载模型参数
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.item_encoder = Encoder(args)
        self.project = Project(args)
        self.args = args
        self.apply(self.init_weights)
        self.cl_loss = NCELoss(args.temperature,'cuda')
        self.loss = 0

    def get_position_embedding(self, sequence):
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        return position_embeddings

    def get_attention_mask(self, sequence):
        attention_mask = (sequence > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(self.device)
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def contrastive_loss(self, mask, feat):
        B, N, D = feat.shape
        feat = feat.view(B * N, D)
        mask = mask.view(B * N)
        feat = feat[mask != 0]
        if feat.size(0) == 0:
            return torch.tensor(0.0, requires_grad=True)
        similarity_matrix = torch.matmul(feat, feat.T)
        similarity_matrix = similarity_matrix
        batch_indices = torch.arange(B).unsqueeze(1).expand(B, N).reshape(-1)
        batch_indices = batch_indices[mask != 0]
        labels = (batch_indices.unsqueeze(0) == batch_indices.unsqueeze(1)).float()
        labels = labels - torch.eye(labels.size(0), device=labels.device)
        logits = similarity_matrix
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        exp_logits = torch.exp(logits) * (labels > 0).float()
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        mean_log_prob_pos = (labels * log_prob).sum(1) / (labels.sum(1) + 1e-12)
        loss = -mean_log_prob_pos
        loss = loss.mean()

        return loss

    def transformer_encoder(self, input_ids, flag=False):
        seq_emb = self.item_embeddings(input_ids.to(self.device))
        #cl_emb = self.project(self.model_cl.transformer_encoder(input_ids))
        cl_emb = self.project(self.feat_embeddings(input_ids))
        pos_emb = self.get_position_embedding(input_ids)
        attention_mask = self.get_attention_mask(input_ids)
        if flag:
            B = seq_emb.size(0)
            seq_emb_1 = seq_emb.reshape(B, -1)
            cl_emb_1 = cl_emb.reshape(B,-1)
            self.loss = self.cl_loss(seq_emb_1, cl_emb_1)
        output = self.item_encoder(
            seq_emb, cl_emb, pos_emb, attention_mask
        )

        return output

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    # def init_weights(self, module):
    #     """ Initialize the weights. """
    #     if isinstance(module, nn.Embedding):
    #         if module == self.model_cl:
    #             mean = module.weight.data.mean()
    #             std = module.weight.data.std()
    #             # 将预训练嵌入重新调整到指定的均值和方差范围
    #             module.weight.data = (module.weight.data - mean) / std  # 标准化
    #             module.weight.data = module.weight.data * self.args.initializer_range + 0.0  # 调整到新范围
    #         else:
    #             # 对其他嵌入层进行正态分布初始化
    #             module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
    #     elif isinstance(module, nn.Linear):
    #         module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
    #         if module.bias is not None:
    #             module.bias.data.zero_()
    #     elif isinstance(module, LayerNorm):
    #         module.bias.data.zero_()
    #         module.weight.data.fill_(1.0)
class NCELoss(nn.Module):
    """
    Eq. (12): L_{NCE}
    """

    def __init__(self, temperature, device):
        super(NCELoss, self).__init__()
        self.device = device
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.temperature = temperature
        self.cossim = nn.CosineSimilarity(dim=-1).to(self.device)

    # #modified based on impl: https://github.com/ae-foster/pytorch-simclr/blob/dc9ac57a35aec5c7d7d5fe6dc070a975f493c1a5/critic.py#L5
    def forward(self, batch_sample_one, batch_sample_two):
        sim11 = torch.matmul(batch_sample_one, batch_sample_one.T) / self.temperature
        sim22 = torch.matmul(batch_sample_two, batch_sample_two.T) / self.temperature
        sim12 = torch.matmul(batch_sample_one, batch_sample_two.T) / self.temperature
        d = sim12.shape[-1]
        sim11[..., range(d), range(d)] = float('-inf')
        sim22[..., range(d), range(d)] = float('-inf')
        raw_scores1 = torch.cat([sim12, sim11], dim=-1)
        raw_scores2 = torch.cat([sim22, sim12.transpose(-1, -2)], dim=-1)
        logits = torch.cat([raw_scores1, raw_scores2], dim=-2)
        labels = torch.arange(2 * d, dtype=torch.long, device=logits.device)
        nce_loss = self.criterion(logits, labels)
        return nce_loss