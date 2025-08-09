import torch
from torch import nn
from torch.nn import Linear, Conv1d, Conv2d, Sequential, ReLU, Dropout, Parameter, Sigmoid
import time
import math

global device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FC0(nn.Module):
    def __init__(self, in_features, out_features):
        super(FC0, self).__init__()
        self.model = Sequential(
            Linear(in_features, 1024),
            ReLU(),
            Linear(1024, 1024),
            ReLU(),
            Linear(1024, 2048),
            ReLU(),
            Linear(2048, 2048),
            ReLU(),
            Linear(2048, 1024),
            ReLU(),
            Linear(1024, out_features),
        )
    def forward(self, x):
        x = self.model(x)
        return x

# Task Head - MLP
class Taskhead(nn.Module):
    def __init__(self, in_features, out_features):
        super(Taskhead, self).__init__()
        self.model = Sequential(
            Linear(in_features, 1024),
            ReLU(),
            Linear(1024, 1024),
            ReLU(),
            Linear(1024, 512),
            ReLU(),
            Linear(512, 256),
            ReLU(),
            Linear(256, 128),
            ReLU(),
            Linear(128, 64),
            ReLU(),
            Linear(64, 32),
            ReLU(),
            Linear(32, 16),
            ReLU(),
            Linear(16, 8),
            ReLU(),
            Linear(8, 4),
            ReLU(),
            Linear(4, out_features),
        )
    def forward(self, x):
        x = self.model(x)
        return x

# Embedding for ddms
class InputEmbeddings_ddms(nn.Module):
    
    def __init__(self, c_in:int, c_out:int, kernel_size:int, stride:int):
        super().__init__()
        input_height, input_width = 11, 17
        pad_bottom = (stride - (input_height % stride)) % stride
        pad_right = (stride - (input_width % stride)) % stride

        self.tokenConv = Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=kernel_size, stride=stride, 
                                   padding=(pad_bottom, pad_right))
        
    def forward(self, x):
        x = self.tokenConv(x)
        return x

# Embedding for APs
class InputEmbeddings_APs(nn.Module):
    
    def __init__(self, c_in:int, c_out:int, kernel_size:int, stride:int):
        super().__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = Conv1d(in_channels=c_in, out_channels=c_out,
                                   kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        
    def forward(self, x):
        x = self.tokenConv(x)
        return x

# Positional Encoding
class PositionalEncoding(nn.Module):

    def __init__(self, d_model:int, seq_len:int, dropout:float) -> None:
        super().__init__() 
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = Dropout(dropout)
        
        d_model_even = d_model if d_model % 2 == 0 else d_model + 1
    
        pe = torch.zeros(seq_len, d_model_even)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model_even, 2).float() * (-math.log(10000.0) / d_model_even))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe[:, :d_model]
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

# LayerNormalization (LN) operation
class LayerNormalization(nn.Module):
    
    def __init__(self, eps:float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = Parameter(torch.ones(1))
        self.bias = Parameter(torch.zeros(1))
    
    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim =True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

# FeedForward Network
class FeedForwardBlock(nn.Module):

    def __init__(self, d_model:int, d_ff:int, dropout:float) -> None:
        super().__init__()
        self.linear_1 = Linear(d_model, d_ff)
        self.dropout = Dropout(dropout)
        self.linear_2 = Linear(d_ff, d_model)
    
    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

# Multi-Head Attention 
class MultiHeadAttentionBlock(nn.Module):
    
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"
        self.d_k = d_model // h
        
        self.w_q = Linear(d_model, d_model)
        self.w_k = Linear(d_model, d_model)
        self.w_v = Linear(d_model, d_model)
        
        self.w_o = Linear(d_model, d_model)
        self.dropout = Dropout(dropout)

    @staticmethod
    def attention(query_h, key_h, value_h, mask, dropout:Dropout):
        d_k = query_h.shape[-1]

        attention_scores = (query_h @ key_h.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9) 
        
        attention_scores = attention_scores.softmax(dim = -1)
        
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        attention_result = attention_scores @ value_h
        
        return (attention_result), attention_scores

    def forward(self, q, k, v, mask=None):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        
        query_h = query.view(query.shape[0], query.shape[1], self.h, self.d_k)
        query_h = query_h.transpose(1, 2)
        
        key_h = key.view(key.shape[0], key.shape[1], self.h, self.d_k)
        key_h = key_h.transpose(1, 2)
        
        value_h = value.view(value.shape[0], value.shape[1], self.h, self.d_k)
        value_h = value_h.transpose(1, 2)
        
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query_h, key_h, value_h, mask, self.dropout)

        x = x.transpose(1,2)
        x = x.contiguous().view(x.shape[0], -1, self.h * self.d_k)  

        output = self.w_o(x)
        return output


# Add & Norm Block
class ResidualConnection(nn.Module):

    def __init__(self, dropout:float) -> None:
        super().__init__()
        self.dropout = Dropout(dropout)
        self.norm = LayerNormalization()
    
    def forward(self, x, sublayer):
        output = x + self.dropout(sublayer(self.norm(x)))
        return output

# Single Encoder Block
class EncoderBlock(nn.Module):
    
    def __init__(self, self_attention_block:MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super(). __init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask=None):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

# Transformer Encoder Block
class Encoder(nn.Module):

    def __init__(self, layers:nn.ModuleList) -> None: 
        super(). __init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)  
        return self.norm(x)

# Linear Layer for APs
class Aps_linear(nn.Module):
    
    def __init__(self, in_features, out_features):
        super(Aps_linear, self).__init__()
        self.model = Sequential(
            Linear(in_features, 128),
            Linear(128, out_features),
            Sigmoid(),
        )
    def forward(self, x):
        x = self.model(x)
        return x

# Projection Layer
class ProjectionLayer(nn.Module):
    
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = Linear(d_model, vocab_size)
    
    def forward(self, x):
        proj_output = torch.log_softmax(self.proj(x), dim = -1)
        return proj_output

# Model Class
class Model(nn.Module):
    
    def __init__(self):
        super(Model, self).__init__()
        self.ddms_patch_embed = InputEmbeddings_ddms(c_in = 3, c_out = 3, kernel_size=2, stride=2)
        self.aps_embed = InputEmbeddings_APs(c_in = 4, c_out = 4, kernel_size=3, stride=1)
        self.glb_token = Parameter(torch.randn(1, 3, 1))
        self.pos_embed = PositionalEncoding(3, 55, 0.1)
        self.encoder_block = EncoderBlock(MultiHeadAttentionBlock(4, 4, 0.1), FeedForwardBlock(4, 256, 0.1), 0.1)
        encoder_blocks = nn.ModuleList([self.encoder_block for _ in range(6)])
        self.encoder = Encoder(layers=encoder_blocks)
        self.kq_linear_data = Aps_linear(9, 9)
        self.kq_linear_channel = Aps_linear(4, 4)
        self.retrievalhead = Taskhead(174, 1)


    def forward(self, BRCS, power, scatter, input_N):
        
        start_time = time.time()
        b_size = input_N.shape[0]
        
        brcs_ch1, brcs_ch2, brcs_ch3, brcs_ch4 = BRCS[:, 0:1, :, :], BRCS[:, 1:2, :, :], BRCS[:, 2:3, :, :], BRCS[:, 3:4, :, :]
        power_ch1, power_ch2, power_ch3, power_ch4 = power[:, 0:1, :, :], power[:, 1:2, :, :], power[:, 2:3, :, :], power[:, 3:4, :, :]
        scatter_ch1, scatter_ch2, scatter_ch3, scatter_ch4 = scatter[:, 0:1, :, :], scatter[:, 1:2, :, :], scatter[:, 2:3, :, :], scatter[:, 3:4, :, :]

        ddms_ch1 = torch.cat([brcs_ch1, power_ch1, scatter_ch1], dim=1)
        ddms_ch2 = torch.cat([brcs_ch2, power_ch2, scatter_ch2], dim=1)
        ddms_ch3 = torch.cat([brcs_ch3, power_ch3, scatter_ch3], dim=1)
        ddms_ch4 = torch.cat([brcs_ch4, power_ch4, scatter_ch4], dim=1)

        ddms_ch1_patch_embed = self.ddms_patch_embed(ddms_ch1)
        ddms_ch2_patch_embed = self.ddms_patch_embed(ddms_ch2)
        ddms_ch3_patch_embed = self.ddms_patch_embed(ddms_ch3)
        ddms_ch4_patch_embed= self.ddms_patch_embed(ddms_ch4)

        ddms_ch1_patch_embed = ddms_ch1_patch_embed.view(b_size, 3, -1)
        ddms_ch2_patch_embed = ddms_ch2_patch_embed.view(b_size, 3, -1)
        ddms_ch3_patch_embed = ddms_ch3_patch_embed.view(b_size, 3, -1)
        ddms_ch4_patch_embed = ddms_ch4_patch_embed.view(b_size, 3, -1)

        ddms_ch1_reg_token = self.glb_token.repeat((b_size, 1, 1))
        ddms_ch2_reg_token = self.glb_token.repeat((b_size, 1, 1))
        ddms_ch3_reg_token = self.glb_token.repeat((b_size, 1, 1))
        ddms_ch4_reg_token = self.glb_token.repeat((b_size, 1, 1))

        ddms_ch1_pat_reg_token = torch.cat([ddms_ch1_patch_embed, ddms_ch1_reg_token], dim=2)
        ddms_ch2_pat_reg_token = torch.cat([ddms_ch2_patch_embed, ddms_ch2_reg_token], dim=2)
        ddms_ch3_pat_reg_token = torch.cat([ddms_ch3_patch_embed, ddms_ch3_reg_token], dim=2)
        ddms_ch4_pat_reg_token = torch.cat([ddms_ch4_patch_embed, ddms_ch4_reg_token], dim=2)

        ddms_ch1_pos_embed = self.pos_embed(ddms_ch1_pat_reg_token.permute(0, 2, 1)).transpose(1, 2)
        ddms_ch2_pos_embed = self.pos_embed(ddms_ch2_pat_reg_token.permute(0, 2, 1)).transpose(1, 2)
        ddms_ch3_pos_embed = self.pos_embed(ddms_ch3_pat_reg_token.permute(0, 2, 1)).transpose(1, 2)
        ddms_ch4_pos_embed = self.pos_embed(ddms_ch4_pat_reg_token.permute(0, 2, 1)).transpose(1, 2)

        ddms_ch1_embed = ddms_ch1_pat_reg_token + ddms_ch1_pos_embed
        ddms_ch1_embed = ddms_ch1_embed.view(b_size, -1)
        ddms_ch2_embed = ddms_ch2_pat_reg_token + ddms_ch2_pos_embed
        ddms_ch2_embed = ddms_ch2_embed.view(b_size, -1)
        ddms_ch3_embed = ddms_ch3_pat_reg_token + ddms_ch3_pos_embed
        ddms_ch3_embed = ddms_ch3_embed.view(b_size, -1)
        ddms_ch4_embed = ddms_ch4_pat_reg_token + ddms_ch4_pos_embed
        ddms_ch4_embed = ddms_ch4_embed.view(b_size, -1)

        ddm_4chs_embed = torch.stack([ddms_ch1_embed, ddms_ch2_embed, ddms_ch3_embed, ddms_ch4_embed], dim=1) 

        encoder_input = ddm_4chs_embed
        encoder_output = self.encoder(encoder_input.permute(0, 2, 1)).transpose(1, 2)

        input_N_embed = self.aps_embed(input_N)
        input_N_data = self.kq_linear_data(input_N_embed)
        input_N_channel = self.kq_linear_channel(input_N_embed.permute(0, 2, 1)).transpose(1, 2)
        input_N_attention = input_N_embed * input_N_data * input_N_channel

        ddms_input_N_feature_cat = torch.cat([encoder_output, input_N_attention], dim=2)
        output = self.retrievalhead(ddms_input_N_feature_cat)

        total_time = time.time() - start_time

        time_info = {
            "Total time": total_time
        }

        return output, time_info
