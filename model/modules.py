"""
Set-Transformer Components: MAB, SAB, ISAB, PMA
Based on the official Set-Transformer repository implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MAB(nn.Module):
    """Multihead Attention Block (MAB)
    
    Core building block for Set-Transformer that performs multihead attention
    between query and key-value pairs.
    
    Args:
        dim_Q: Dimension of query
        dim_K: Dimension of key
        dim_V: Dimension of value (output)
        num_heads: Number of attention heads
        ln: Whether to use LayerNorm
    """
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K, key_mask=None):
        """
        Args:
            Q: Query tensor of shape (batch_size, n_queries, dim_Q)
            K: Key-Value tensor of shape (batch_size, n_keys, dim_K)
            
        Returns:
            Output tensor of shape (batch_size, n_queries, dim_V)
        """
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        # A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        d_k = self.dim_V // self.num_heads
        attn_logits = Q_.bmm(K_.transpose(1,2)) / math.sqrt(max(d_k, 1))

        #apply mask if provided
        if key_mask is not None:
            #key mask: (batch, n_leys) -> expand to (batch * heads, 1, n_keys)
            mask = (~key_mask).unsqueeze(1) # true where invalid
            mask = mask.repeat(self.num_heads, 1, 1) #repeat for each head
            attn_logits = attn_logits.masked_fill(mask, -1e-9)

        A = torch.softmax(attn_logits, dim=2)

        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class SAB(nn.Module):
    """Set Attention Block (SAB)
    
    Self-attention variant of MAB where Q and K are the same.
    
    Args:
        dim_in: Input dimension
        dim_out: Output dimension
        num_heads: Number of attention heads
        ln: Whether to use LayerNorm
    """
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X, mask=None):
        """
        Args:
            X: Input tensor of shape (batch_size, n_points, dim_in)
            
        Returns:
            Output tensor of shape (batch_size, n_points, dim_out)
        """
        return self.mab(X, X, key_mask=mask)


class ISAB(nn.Module):
    """Induced Set Attention Block (ISAB)
    
    Efficient attention block using learned inducing points to reduce
    computational complexity from O(n²) to O(mn) where m << n.
    
    Args:
        dim_in: Input dimension
        dim_out: Output dimension
        num_heads: Number of attention heads
        num_inds: Number of inducing points
        ln: Whether to use LayerNorm
    """
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X, mask=None):
        """
        Args:
            X: Input tensor of shape (batch_size, n_points, dim_in)
            
        Returns:
            Output tensor of shape (batch_size, n_points, dim_out)
        """
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X, key_mask=mask)
        return self.mab1(X, H)


class PMA(nn.Module):
    """Pooling by Multihead Attention (PMA)
    
    Aggregates set elements into a fixed number of seed vectors using attention.
    
    Args:
        dim: Feature dimension
        num_heads: Number of attention heads
        num_seeds: Number of output seed vectors
        ln: Whether to use LayerNorm
    """
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        """
        Args:
            X: Input tensor of shape (batch_size, n_points, dim)
            
        Returns:
            Output tensor of shape (batch_size, num_seeds, dim)
        """
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)