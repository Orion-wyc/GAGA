import torch
import torch.nn as nn

from modules import embedding


class TransformerEncoderNet(nn.Module):
    def __init__(self, feat_dim, emb_dim, n_classes, n_hops, n_relations,
                 n_heads, dim_feedforward, n_layers, dropout=0.1, agg_type='cat'):
        r"""Transformer encoder based on torch built-in modules.
        Currently, the graph transformer is based on the implementation in PyTorch.
            # todo (yuchen): re-implement based on ViT 
            # Reference <https://github.com/lucidrains/vit-pytorch>
        Parameters
        ----------
        feat_dim : int
            Input feature size; i.e., number of  dimensions of the raw input feature.
        emb_dim : int
            Hidden size of all learning embeddings and hidden vectors. 
            (deotes by E)
        n_classes : int
            Number of classes. 
            (deotes by C)
        n_hops : int
            Number of hops (mulit-hop neighborhood information). (deotes by K)
        n_relations : int
            Number of relations. 
            (deotes by R)
        n_heads : int
            Number of heads in MultiHeadAttention module.
        dim_feedforward : int
        n_layers : int
            Number of encoders layers. 
        dropout: float
            Dropout rate on feature. Default=0.1.
        agg_type: str
            Cross-relation aggregation type, including 'cat' and 'mean'.
        
        Return : torch.Tensor
            Final representation of target node(s). 
            Shape=(N, R \times E)     
        """ 
        super(TransformerEncoderNet, self).__init__()

        # encoder that provides hop, relation and group encodings
        self.feat_encoder = embedding.CustomEncoder(feat_dim=feat_dim,
                                                    emb_dim=emb_dim, n_relations=n_relations,
                                                    n_hops=n_hops, dropout=dropout,
                                                    n_classes=n_classes)

        # define transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(emb_dim, n_heads, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)

        # cross-relation aggregation type (Figure 2 \ding{206})
        if agg_type == 'cat':
            proj_emb_dim = emb_dim * n_relations
        elif agg_type == 'mean':
            proj_emb_dim = emb_dim

        # the MLP (Figure 2 \ding{206})
        self.projection = nn.Sequential(nn.Linear(proj_emb_dim, n_classes))
        self.dropout = nn.Dropout(dropout)

        self.emb_dim = emb_dim
        self.n_hops = n_hops
        self.n_relations = n_relations
        self.n_classes = n_classes
        self.agg_type = agg_type

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def cross_relation_agg(self, out):
        r"""Aggregate target node's outputs under all relations.
        Parameters
        ----------
        out : torch.Tensor
            The output tensor of Transformer Encoder.
            Shape = (S, N, E)
        
        """
        # todo (yuchen): provide ['cat', 'mean', 'weight', 'atten'] aggregators
        device = out.device
        n_tokens = out.shape[0]

        # extract output vector(s) of the target node under each relation
        block_len = 1 + self.n_hops * (self.n_classes + 1)
        indices = torch.arange(0, n_tokens, block_len, dtype=torch.int64).to(device)

        # (n_relations, N, E)
        mr_feats = torch.index_select(out, dim=0, index=indices)
        if self.agg_type == 'cat':
            #  (N,E) tuple_len = n_relations
            mr_feats = torch.split(mr_feats, 1, dim=0)

            # (N,n_relations*E)
            agg_feats = torch.cat(mr_feats, dim=2).squeeze()

        elif self.agg_type == 'mean':
            # (N,E)
            agg_feats = torch.mean(mr_feats, dim=0)

        return agg_feats

    def forward(self, src_emb, src_mask=None):
        r"""
        Parameters
        ----------
        src_emb : Tensor
            Input feature sequence. Shape (S, N, E)
        src_mask : ?
            Currently useless.
        """
        # input feature sequence (N,S,E)->(S,N,E)
        src_emb = torch.transpose(src_emb, 1, 0)

        # encoding in Section 4.3
        out = self.feat_encoder(src_emb)

        # transformer encoder
        out = self.transformer_encoder(out, src_mask)

        # cross-relation aggregation, out(S,N,E)-->(N,E)
        out = self.cross_relation_agg(out)

        # prediction
        out = self.projection(out)

        return out 
