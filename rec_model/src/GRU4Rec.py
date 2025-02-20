
import torch
from torch import nn
from torch.nn.init import xavier_uniform_, xavier_normal_

class GRU(nn.Module):

    def __init__(self, args):
        super(GRU, self).__init__(args)
        self.emb_dropout = nn.Dropout(args.dropout_prob)
        self.gru_layers = nn.GRU(
            input_size=args.hidden_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
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
