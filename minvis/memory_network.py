import logging
import math
from typing import Tuple
import einops

import torch
from torch import nn
from torch.nn import functional as F

class MemoryNetwork(nn.Module):
    def __init__(self, query_dim, appearance_dim):
        super().__init__()
        self.temporal_aggregation = nn.RNNCell(appearance_dim, appearance_dim)

        self.q_appearance_proj = nn.Linear(appearance_dim, appearance_dim)
        self.k_appearance_proj = nn.Linear(appearance_dim, appearance_dim)

        self.q_query_proj = nn.Linear(query_dim, query_dim)
        self.k_query_proj = nn.Linear(query_dim, query_dim)

        self.k_query = None
        self.k_appearance = None
        
    def init_network(self, k_query, k_appearance):
        self.k_query = k_query
        self.k_appearance = self.temporal_aggregation(k_appearance.flatten(0,1))

    def get(self, q_query, q_appearance):
        '''
        q_appearance: (B, Q, D)
        q_query: (B, Q, C)
        '''
        B, Q, _ = q_query.shape

        q_appearance = self.q_appearance_proj(q_appearance)
        q_query = self.q_query_proj(q_query)

        k_appearance = self.k_appearance_proj(self.k_appearance.reshape(B, Q, -1)) 
        k_query = self.k_query_proj(self.k_query) 

        query_similarity = torch.einsum('bqc, bkc -> bqk', q_query, k_query)#.softmax(-1) # B, Q, Q
        appearance_map = torch.einsum('bqc, bkc -> bqk', q_appearance, k_appearance) # B, Q, Q

        similarity = query_similarity * appearance_map
        
        return similarity
    
    def loss(self, similarity, label):
        '''
        similarity: (B, Q, Q)
        '''
        # loss = F.binary_cross_entropy_with_logits(similarity, label.float())
        loss = (torch.abs(similarity - label.float())).mean(dim=(1,2)).sum()

        return loss
    
    def update(self, k_query, k_appearance):
        '''
        k_appearance: (B, Q, D)
        k_query: (B, Q, C)
        '''
        B, Q, _ = k_query.shape

        self.k_query = k_query
        self.k_appearance = self.temporal_aggregation(k_appearance.flatten(0,1), self.k_appearance)
