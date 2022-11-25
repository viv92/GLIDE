### Utility implementing the text encoder (transformer) for GLIDE

import math
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# utility function to create N copies of a module as a list (note: not sequential)
def clones(module, N):
    return nn.ModuleList([deepcopy(module) for _ in range(N)])

# utility function to create upper triangular mask for decoder masked attention
def subsequent_mask(mask_shape):
    batch_size, max_seq_len = mask_shape
    mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).type(torch.uint8) # mask.shape = [max_seq_len, max_seq_len]
    mask = mask.unsqueeze(0).expand(batch_size, max_seq_len, max_seq_len) # mask.shape = [batch_size, max_seq_len, max_seq_len]
    return mask == 0 # False elements are masked

# utility function to create mask over pad tokens
def pad_mask(keys, pad_token):
    batch_size, max_seq_len = keys.shape
    mask = keys.eq(pad_token).unsqueeze(1) # mask.shape: [batch_size, 1, max_seq_len]
    mask = mask.expand(batch_size, max_seq_len, max_seq_len) # mask.shape: [batch_size, max_seq_len, max_seq_len]
    return mask == 0 # False elements are masked

# class for caption embeddings
class CaptionEmbeddings(nn.Module):
    def __init__(self, vocab_size, max_seq_len, d_model, dropout, device):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.device = device
    def forward(self, x):
        batch_size, max_seq_len = x.shape[0], x.shape[1]
        tok_emb = self.tok_emb(x) # tok_emb.shape: [batch_size, max_seq_len, d_model]
        positions = torch.arange(max_seq_len).to(self.device)
        positions = positions.unsqueeze(0)
        positions = positions.expand(batch_size, -1)
        pos_emb = self.pos_emb(positions)
        final_emb = self.dropout( self.norm(tok_emb + pos_emb) )
        final_emb = final_emb * math.sqrt(self.d_model)
        return final_emb


# class implementing the feed forward block (used for each encoder / decoder layer - after the multihead attention block)
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.w2(self.dropout(self.w1(x).relu()))

# class implementing multi head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, d_k, d_v, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.output = nn.Linear(d_v * n_heads, d_model, bias=False)
        self.attn_weights = None # placeholder to store attention weights (used to visualize attention matrices)
        self.dropout = nn.Dropout(dropout)

    # function to calculate (masked or unmasked) multihead attention
    def forward(self, key, query, value, mask=None): # can be used for both (unmasked) encoder attention and (masked) decoder attention
        # key.shape: [batch_size, seq_len, d_model]; mask.shape: [batch_size, seq_len, seq_len]
        # project key, query, value and reshape into multiple heads
        batch_size = key.shape[0]
        # (batch_size, seq_len, d_model) -proj-> (batch_size, seq_len, proj_dim) -view-> (batch_size, seq_len, n_heads, d_k) -transpose-> (batch_size, n_heads, seq_len, d_k)
        proj_key = self.W_K(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        proj_query = self.W_Q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        proj_value = self.W_V(value).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        # expand mask for n_heads
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # mask.shape: [batch_size, n_heads, seq_len, seq_len]
        # calculate attention
        attn_multihead, attn_weights = self.scaled_dotprod_attn(proj_key, proj_query, proj_value, mask, self.dropout)
        attn_multihead = attn_multihead.transpose(1, 2) # attn_multihead.shape: [batch_size, seq_len, n_heads, d_v]
        attn_multihead = torch.flatten(attn_multihead, start_dim=-2, end_dim=-1) # attn_multihead.shape: [batch_size, seq_len, n_heads * d_v]
        attn_multihead = self.output(attn_multihead) # attn_multihead.shape: [batch_size, seq_len, d_model]
        self.attn_weights = attn_weights
        return attn_multihead

    # function to calculate scaled dot product attention for one head
    def scaled_dotprod_attn(self, key, query, value, mask=None, dropout=None): # key.shape: [batch_size, n_heads, seq_len, d_k]
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k) # attn_scores.shape: [batch_size, n_heads, seq_len, seq_len]
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask==0, -1e9)
        attn_weights = attn_scores.softmax(dim=-1) # attn_weights.shape: [batch_size, n_heads, seq_len, seq_len]
        if dropout is not None:
            attn_weights = dropout(attn_weights)
        attn_vector = torch.matmul(attn_weights, value) # attn_vector.shape: [batch_size, n_heads, seq_len, d_v]
        return attn_vector, attn_weights

# class implementing Layer Normalization
class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.a = nn.Parameter(torch.ones(dim))
        self.b = nn.Parameter(torch.zeros(dim))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * ( (x - mean) / (std + self.eps) ) + self.b

# class implementing residual + normalization connection - takes in any block and applies a residual connection around it + a layer normalization on top
class SublayerConnection(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, sublayer): # sublayer can be any functional block
        return x + self.dropout( sublayer( self.norm(x) ) )

# class implementing a single encoder layer
# each encoder layer has two blocks: 1. (self) multihead attention 2. feed_forward; with sublayer connection around each
class EncoderLayer(nn.Module):
    def __init__(self, self_attn, feed_forward, dim, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayers = clones(SublayerConnection(dim, dropout), 2) # one for self_attn block and other for feed_forward block
    def forward(self, x, mask):
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, mask)) # x.shape: [batch_size, seq_len, d_model]
        x = self.sublayers[1](x, self.feed_forward) # x.shape: [batch_size, seq_len, d_model]
        return x

# class implementing the entire encoder block = stacked encoder layers
class Encoder(nn.Module):
    def __init__(self, layer, N, dim):
        super().__init__()
        self.layers = clones(layer, N)
        self.final_norm = LayerNorm(dim)
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.final_norm(x)

# class implemeting the entire text_encoder model
class Text_Encoder(nn.Module):
    def __init__(self, transformer_encoder, caption_embeddings_model, d_model, pad_token, device):
        super().__init__()
        self.transformer_encoder = transformer_encoder
        self.caption_embeddings = caption_embeddings_model
        self.eos_proj = nn.Linear(d_model, d_model, bias=False)
        self.pad_token = pad_token
        self.device = device

    def get_embeddings_and_mask(self, captions):
        # get caption mask
        cap_pad_mask = pad_mask(captions, self.pad_token) # pad mask for captions
        cap_sub_mask = subsequent_mask(captions.shape).to(self.device)
        cap_mask = torch.logical_and( cap_pad_mask, cap_sub_mask ) # add subsequent mask for captions
        # get caption embeddings
        cap_embs = self.caption_embeddings(captions) # shape: [batch_size, max_seq_len, d_model]
        return cap_embs, cap_mask

    def encode(self, src, src_mask):
        encoder_out = self.transformer_encoder(src, src_mask)
        return encoder_out

    def forward(self, captions):
        cap_embs, cap_mask = self.get_embeddings_and_mask(captions)
        enc_out = self.encode(cap_embs, cap_mask) # enc_out.shape: [batch_size, seq_len, d_model]
        eos_out = self.eos_proj(enc_out[:, -1]) # eos_out.shape: [batch_size, d_model]
        return enc_out, eos_out

# caller function to instantiate the text encoder model, using the defined hyperparams as input
def init_text_encoder(vocab_size, max_seq_len, d_model, d_k, d_v, n_heads, n_layers, d_ff, dropout, pad_token, device):
    attn = MultiHeadAttention(n_heads, d_model, d_k, d_v, dropout) # multi head attention block
    ff = FeedForward(d_model, d_ff, dropout) # feed forward block for each encoder / decoder block
    caption_embeddings_model = CaptionEmbeddings(vocab_size, max_seq_len, d_model, dropout, device)
    encoder_layer = EncoderLayer(deepcopy(attn), deepcopy(ff), d_model, dropout) # single encoder layer
    encoder = Encoder(encoder_layer, n_layers, d_model) # encoder = stacked encoder layers
    model = Text_Encoder(encoder, caption_embeddings_model, d_model, pad_token, device) # the text_encoder model used for GLIDE
    # initialize params - Xavier initialization
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
