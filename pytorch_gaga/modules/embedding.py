import torch
import torch.nn as nn


class CustomEncoder(nn.Module):
    def __init__(self, feat_dim, emb_dim, n_classes, n_hops, n_relations, dropout=0.1):
        r"""Hop, relation and group encoding strategies.
        This part conresponds to Section 4.3 in our paper.

        The shape of the output is (S, N, E), where S is input sequence length 
        ($S = n_relations \times (n_hops \times (n_classes + 1) + 1)$),
        N is the batch size, E is the output embedding size.

        Parameters
        ----------
        feat_dim: int
             Input feature size; i.e., number of  dimensions of the raw input feature.
        emb_dim: int
            Hidden size; i.e., number of dimensions of hidden embeddings.
        n_classes: int
            Number of classes; e.g., fraud detection only involves 2 classes.
        n_hops: int
            Number of hops/layers. 
        n_relations: int
            Number of relations.
        dropout: float
            Dropout rate on feature. Default=0.1.
        
        Return : torch.Tensor
            Feature sequence with encoding strategies as the input of transformer 
            encoder. 
        """
        super(CustomEncoder, self).__init__()
        self.hop_embedding = HopEmbedding(n_hops + 1, emb_dim)
        self.relation_embedding = RelationEmbedding(n_relations, emb_dim)
        self.group_embedding = GroupEmbedding(n_classes + 1, emb_dim)

        # linear  projection
        self.MLP = nn.Sequential(nn.Linear(feat_dim, emb_dim),
                                 nn.ReLU())

        self.dropout = nn.Dropout(dropout)

        self.feat_dim = feat_dim
        self.emb_dim = emb_dim
        self.n_hops = n_hops
        self.n_relations = n_relations
        self.n_classes = n_classes

        # number of groups under single relation
        self.n_groups = n_classes + 1

        # input sequence length under single relartion
        self.base_seq_len = n_hops * (n_classes + 1) + 1

    def forward(self, x):
        # out(S, N, E), broadcast automatically in torch
        device = x.device

        # the device of indices should consist with the learnable embeddings
        hop_idx = torch.arange(self.n_hops + 1, dtype=torch.int64).to(device)
        rel_idx = torch.arange(self.n_relations, dtype=torch.int64).to(device)
        grp_idx = torch.arange(self.n_groups, dtype=torch.int64).to(device)
        
        # -------- HOP ENCODING STRATEGY --------
        # hop_emb (n_hops+1, E)
        hop_emb = self.hop_embedding(hop_idx)
        # hop_emb (1, E) of the target node v
        center_hop_emb = hop_emb[0].unsqueeze(0)
        # hop_emb (n_hops * n_groups) for all hops
        hop_emb_list = [center_hop_emb]
        for i in range(1, self.n_hops + 1):
            hop_emb_list.append(hop_emb[i].repeat(self.n_groups, 1))
        # (S,E)  S = n_realtions*base_seq_len
        hop_emb = torch.cat(hop_emb_list, dim=0).repeat(self.n_relations, 1)

        # -------- RELATION ENCODING STRATEGY --------
        # rel_emb (n_relations,E)
        rel_emb = self.relation_embedding(rel_idx)
        # (S, E)  S = n_realtions*base_seq_len
        rel_emb = rel_emb.repeat(1, self.base_seq_len).view(-1, self.emb_dim)
        
        # -------- GROUP ENCODING STRATEGY --------
        # grp_emb (n_groups, E)
        # grp_emb for the target node (* unknown group)
        grp_emb = self.group_embedding(grp_idx)
        # (1,E)
        center_grp_emb = grp_emb[-1].unsqueeze(0)
        # (n_hop*n_groups, E)
        hop_grp_emb = grp_emb.repeat(self.n_hops, 1)
        # (S,E) S = n_realtions*base_seq_len
        grp_emb = torch.cat((center_grp_emb, hop_grp_emb), dim=0).repeat(self.n_relations, 1)

        # linear projection
        out = self.MLP(x)
        
        # broadcast x: [S, N, E] + (S,1,E) 
        out = out + hop_emb.unsqueeze(1) + rel_emb.unsqueeze(1) + grp_emb.unsqueeze(1)
        
        out = self.dropout(out)

        return out


class HopEmbedding(nn.Embedding):
    def __init__(self, max_len, emb_dim=128):
        """Hop Embeddings.

        Parameters
        ----------
        max_len: int
            Number of learnable embeddings.
        emb_dim: int
            Embedding size, i.e., number of dimensions of learnable embeddings.
        """
        super(HopEmbedding, self).__init__(max_len, emb_dim)


class RelationEmbedding(nn.Embedding):
    def __init__(self, max_len: int, emb_dim=128):
        """Relation Embeddings.
        
        Parameters
        ----------
        max_len: int
            Number of learnable embeddings.
        emb_dim: int
            Embedding size, i.e., number of dimensions of learnable embeddings.
        """
        super(RelationEmbedding, self).__init__(max_len, emb_dim)


class GroupEmbedding(nn.Embedding):
    def __init__(self, max_len: int, emb_dim=128):
        """Group Embeddings.
        
        Parameters
        ----------
        max_len: int
            Number of learnable embeddings.
        emb_dim: int
            Embedding size, i.e., number of dimensions of learnable embeddings.
        """
        super(GroupEmbedding, self).__init__(max_len, emb_dim)
