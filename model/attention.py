from einops import rearrange
from inspect import isfunction

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.attention_processor import Attention, AttnProcessor

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)

class MaskedSelfAttention(nn.Module):
    def __init__(self, query_dim, heads=8, dim_head=64, dropout=0.,):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout) )
    
    def forward(self, x, attention_masks):
        q = self.to_q(x) # B*N*(H*C)
        k = self.to_k(x) # B*N*(H*C)
        v = self.to_v(x) # B*N*(H*C)

        B, N, HC = q.shape 
        H = self.heads
        C = HC // H 

        q = q.view(B,N,H,C).permute(0,2,1,3) # B*H*N*C
        k = k.view(B,N,H,C).permute(0,2,1,3) # B*H*N*C
        v = v.view(B,N,H,C).permute(0,2,1,3) # B*H*N*C

        q = q.reshape(B*H,N,C) # (B*H)*N*C
        k = k.reshape(B*H,N,C) # (B*H)*M*C
        v = v.reshape(B*H,N,C) # (B*H)*M*C

        sim = torch.einsum('b i c, b j c -> b i j', q, k) * self.scale  # (B*H)*N*N
        sim = sim.view(B,H,N,N).masked_fill(attention_masks <= 0.0, -np.inf).view(B*H,N,N) # -np.inf

        attn = sim.softmax(dim=-1) # (B*H)*N*N
        out = torch.einsum('b i j, b j c -> b i c', attn, v) # (B*H)*N*C
        out = out.view(B,H,N,C).permute(0,2,1,3).reshape(B,N,(H*C)) # B*N*(H*C)
        return self.to_out(out)

class CustomAttnProcessor(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.name = args['name']
        self.is_cma = False
        if not args['mlp_dim'] == None:
            self.is_cma = True
            query_dim = args['mlp_dim']
            self.linear = nn.Linear(768, query_dim)
            self.norm1 = nn.LayerNorm(query_dim)
            self.norm2 = nn.LayerNorm(query_dim)
            self.norm3 = nn.LayerNorm(query_dim)
            self.ff = FeedForward(query_dim, glu=True)
            self.cma = MaskedSelfAttention(query_dim)
            self.register_parameter('alpha_attn', nn.Parameter(torch.tensor(0.)) )
            self.register_parameter('alpha_dense', nn.Parameter(torch.tensor(0.)) )

    """
    Args:
        Q : hidden_states: B * (H * W) * C
        K V :encoder_hidden_states: B * L * C = B * 77 * 768
    """
    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        object_embeddings=None,
        object_attention_masks=None
    ):
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            if self.is_cma:
                n_visual = hidden_states.shape[1]
                object_embeddings = self.linear(object_embeddings)
                attention_output = self.cma(self.norm1(torch.cat([hidden_states, object_embeddings], dim=1)), object_attention_masks)
                hidden_states = hidden_states + torch.tanh(self.alpha_attn) * attention_output[:, 0:n_visual,:]
                hidden_states = hidden_states + torch.tanh(self.alpha_dense) * self.ff( self.norm2(hidden_states) ) 
                hidden_states = self.norm3(hidden_states) 
        
        # custom_args+=1
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # To do Mask Attention
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)

        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states

attention_info_dict = {
    "down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor" : {'type':  'SA', 'dim': [1, 4096, 320]},
    "down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor" : {'type':  'CA', 'dim': [1, 4096, 320]},
    "down_blocks.0.attentions.1.transformer_blocks.0.attn1.processor" : {'type':  'SA', 'dim': [1, 4096, 320]}, 
    "down_blocks.0.attentions.1.transformer_blocks.0.attn2.processor" : {'type':  'CA', 'dim': [1, 4096, 320]}, 
    "down_blocks.1.attentions.0.transformer_blocks.0.attn1.processor" : {'type':  'SA', 'dim': [1, 1024, 640]},  
    "down_blocks.1.attentions.0.transformer_blocks.0.attn2.processor" : {'type':  'CA', 'dim': [1, 1024, 640]},  
    "down_blocks.1.attentions.1.transformer_blocks.0.attn1.processor" : {'type':  'SA', 'dim': [1, 1024, 640]},  
    "down_blocks.1.attentions.1.transformer_blocks.0.attn2.processor" : {'type':  'CA', 'dim': [1, 1024, 640]},  
    "down_blocks.2.attentions.0.transformer_blocks.0.attn1.processor" : {'type':  'SA', 'dim': [1, 256, 1280]},  
    "down_blocks.2.attentions.0.transformer_blocks.0.attn2.processor" : {'type':  'CA', 'dim': [1, 256, 1280]},  
    "down_blocks.2.attentions.1.transformer_blocks.0.attn1.processor" : {'type':  'SA', 'dim': [1, 256, 1280]},  
    "down_blocks.2.attentions.1.transformer_blocks.0.attn2.processor" : {'type':  'CA', 'dim': [1, 256, 1280]},  
    "mid_block.attentions.0.transformer_blocks.0.attn1.processor" : {'type':  'SA', 'dim': [1, 64, 1280]},      
    "mid_block.attentions.0.transformer_blocks.0.attn2.processor" : {'type':  'CA', 'dim': [1, 64, 1280]},     
    "up_blocks.1.attentions.0.transformer_blocks.0.attn1.processor" : {'type':  'SA', 'dim': [1, 256, 1280]},  
    "up_blocks.1.attentions.0.transformer_blocks.0.attn2.processor" : {'type':  'CA', 'dim': [1, 256, 1280]},  
    "up_blocks.1.attentions.1.transformer_blocks.0.attn1.processor" : {'type':  'SA', 'dim': [1, 256, 1280]},  
    "up_blocks.1.attentions.1.transformer_blocks.0.attn2.processor" : {'type':  'CA', 'dim': [1, 256, 1280]},  
    "up_blocks.1.attentions.2.transformer_blocks.0.attn1.processor" : {'type':  'SA', 'dim': [1, 256, 1280]},  
    "up_blocks.1.attentions.2.transformer_blocks.0.attn2.processor" : {'type':  'CA', 'dim': [1, 256, 1280]}, 
    "up_blocks.2.attentions.0.transformer_blocks.0.attn1.processor" : {'type':  'SA', 'dim': [1, 1024, 640]},  
    "up_blocks.2.attentions.0.transformer_blocks.0.attn2.processor" : {'type':  'CA', 'dim': [1, 1024, 640]},  
    "up_blocks.2.attentions.1.transformer_blocks.0.attn1.processor" : {'type':  'SA', 'dim': [1, 1024, 640]},  
    "up_blocks.2.attentions.1.transformer_blocks.0.attn2.processor" : {'type':  'CA', 'dim': [1, 1024, 640]}, 
    "up_blocks.2.attentions.2.transformer_blocks.0.attn1.processor" : {'type':  'SA', 'dim': [1, 1024, 640]}, 
    "up_blocks.2.attentions.2.transformer_blocks.0.attn2.processor" : {'type':  'CA', 'dim': [1, 1024, 640]}, 
    "up_blocks.3.attentions.0.transformer_blocks.0.attn1.processor" : {'type':  'SA', 'dim': [1, 4096, 320]}, 
    "up_blocks.3.attentions.0.transformer_blocks.0.attn2.processor": {'type':  'CA', 'dim': [1, 4096, 320]}, 
    "up_blocks.3.attentions.1.transformer_blocks.0.attn1.processor" : {'type':  'SA', 'dim': [1, 4096, 320]}, 
    "up_blocks.3.attentions.1.transformer_blocks.0.attn2.processor" : {'type':  'CA', 'dim': [1, 4096, 320]}, 
    "up_blocks.3.attentions.2.transformer_blocks.0.attn1.processor" : {'type':  'SA', 'dim': [1, 4096, 320]}, 
    "up_blocks.3.attentions.2.transformer_blocks.0.attn2.processor" : {'type':  'CA', 'dim': [1, 4096, 320]} 
}

def register_attention_control(unet):
    attn_procs = {}    
    args = {'mlp_dim' : None}
    for name in unet.attn_processors.keys():
        args['name'] = name
        args['mlp_dim'] = None
        if attention_info_dict[name]['type'] == 'CA' and attention_info_dict[name]['dim'][1] == 4096:
            args['mlp_dim'] = attention_info_dict[name]['dim'][-1]
        attn_procs[name] = CustomAttnProcessor(args)
    unet.set_attn_processor(attn_procs)