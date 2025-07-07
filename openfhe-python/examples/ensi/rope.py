
import pycuda.driver as cuda  
import pycuda.autoinit  
from pycuda.compiler import SourceModule  
import pycuda.gpuarray as gpuarray  
import numpy as np  
import time  
from openfhe import *  # Import OpenFHE library
import torch  
import torch.nn as nn 
import random

class RotaryEmbedding(nn.Module):  
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):  
        super().__init__()  
        self.scaling_factor = scaling_factor  
        self.dim = dim  
        self.max_position_embeddings = max_position_embeddings  
        self.base = base  
        
        # Calculate inverse frequency  
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))  
        self.register_buffer("inv_freq", inv_freq)  
        
        # Precompute and cache cos and sin values  
        self.max_seq_len_cached = max_position_embeddings  
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)  
        t = t / self.scaling_factor  
        freqs = torch.outer(t, self.inv_freq)  
        # Duplicate frequencies to match dimensions  
        emb = torch.cat((freqs, freqs), dim=-1)  
        # Cache cos and sin values directly as float16  
        self.register_buffer("_cos_cached", emb.cos().half(), persistent=False)  
        self.register_buffer("_sin_cached", emb.sin().half(), persistent=False)  

    @property  
    def sin_cached(self):  
        logger.warning_once(  
            "The sin_cached attribute will be removed in 4.39. Bear in mind that its contents changed in v4.38. Use "  
            "the forward method of RoPE from now on instead. It is not used in the `BitnetAttention` class"  
        )  
        return self._sin_cached  

    @property  
    def cos_cached(self):  
        logger.warning_once(  
            "The cos_cached attribute will be removed in 4.39. Bear in mind that its contents changed in v4.38. Use "  
            "the forward method of RoPE from now on instead. It is not used in the `BitnetAttention` class"  
        )  
        return self._cos_cached  

    @torch.no_grad()  
    def forward(self, x, position_ids):  
        # x: [bs, num_attention_heads, seq_len, head_size]  
        # position_ids: [bs, seq_len]  
        
        # Ensure position_ids are within valid range  
        position_ids = position_ids.clamp(0, self.max_seq_len_cached - 1)  

        # Get corresponding cos and sin values directly from cache  
        cos = self._cos_cached[position_ids]  # [bs, seq_len, dim]  
        sin = self._sin_cached[position_ids]  # [bs, seq_len, dim]  
        
        # Return as float16 directly, regardless of input x's type  
        return cos, sin       


def cipher_rotate_half(cc, keys, enc_col, seq):  
    """  
    Ciphertext version of rotate_half implementation - processes a single column vector  
    
    Parameters:  
    cc: Crypto context  
    keys: Key pair  
    enc_col: Single encrypted column vector  
    seq: Sequence length  
    
    Returns:  
    result: Rotated encrypted column vector  
    """  
    # print("\n=== Executing ciphertext rotate_half operation ===")  

    # Create two plaintext vectors for multiplication  
    neg_pattern = [-1 if i % 2 == 0 else 0 for i in range(seq)]  
    pos_pattern = [1 if i % 2 == 1 else 0 for i in range(seq)]  

    # Convert plaintext vectors to CKKS plaintexts  
    neg_plain = cc.MakeCKKSPackedPlaintext(neg_pattern, 1, 0, None, seq)  
    pos_plain = cc.MakeCKKSPackedPlaintext(pos_pattern, 1, 0, None, seq)  

    # 1. Rotate and process the original input  
    # Rotate right by 1 position  
    rotated_right = cc.EvalRotate(enc_col, 1)  
    # Multiply with negative pattern  
    mult_neg = cc.EvalMult(rotated_right, neg_plain)  

    # 2. Perform reverse rotation and processing on original input  
    # Rotate left by 1 position  
    rotated_left = cc.EvalRotate(enc_col, -1)  
    # Multiply with positive pattern  
    mult_pos = cc.EvalMult(rotated_left, pos_plain)  

    # 3. Add the two results  
    result = cc.EvalAdd(mult_neg, mult_pos)  

    return result  


def cipher_apply_rotary_pos_emb(cc, keys, q_enc_cols, k_enc_cols, cos_cols, sin_cols, seq, heads, dim):  
    """
    Ciphertext version of apply_rotary_pos_emb implementation
    
    Parameters:
    cc: Crypto context
    keys: Key pair
    q_enc_cols: Encrypted query columns
    k_enc_cols: Encrypted key columns
    cos_cols: Cosine values for rotary embeddings
    sin_cols: Sine values for rotary embeddings
    seq: Sequence length
    heads: Number of attention heads
    dim: Dimension per head
    
    Returns:
    q_embed_cols, k_embed_cols: Encrypted query and key columns with rotary embeddings applied
    """
    # Convert cos and sin to CKKS plaintexts  
    cos_plains = []  
    sin_plains = []  
    for d in range(dim):  
        cos_plains.append(cc.MakeCKKSPackedPlaintext(cos_cols[d], 1, 0, None, seq))  
        sin_plains.append(cc.MakeCKKSPackedPlaintext(sin_cols[d], 1, 0, None, seq))  

    # Create result lists  
    q_embed_cols = []  
    k_embed_cols = []  

    # Process each head and dimension  
    total_dim = heads * dim  
    for i in range(total_dim):  
        h = i // dim  # Current head index  
        d = i % dim   # Current dimension index  

        # Use cos/sin values corresponding to dimension d  
        cos_plain = cos_plains[d]  
        sin_plain = sin_plains[d]  

        # Process query  
        # 1. Calculate q * cos  
        q_cos = cc.EvalMult(q_enc_cols[i], cos_plain)  

        # 2. Calculate rotate_half(q) * sin  
        q_rotated = cipher_rotate_half(cc, keys, q_enc_cols[i], seq)  
        q_rot_sin = cc.EvalMult(q_rotated, sin_plain)  

        # 3. Add the two parts  
        q_embed_cols.append(cc.EvalAdd(q_cos, q_rot_sin))  

        # Process key (same process as query)  
        # 1. Calculate k * cos  
        k_cos = cc.EvalMult(k_enc_cols[i], cos_plain)  

        # 2. Calculate rotate_half(k) * sin  
        k_rotated = cipher_rotate_half(cc, keys, k_enc_cols[i], seq)  
        k_rot_sin = cc.EvalMult(k_rotated, sin_plain)  

        # 3. Add the two parts  
        k_embed_cols.append(cc.EvalAdd(k_cos, k_rot_sin))  

    return q_embed_cols, k_embed_cols
