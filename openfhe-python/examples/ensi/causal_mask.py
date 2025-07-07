
from openfhe import *  
import time  
import numpy as np   
import random

def apply_causal_mask_cipher(cc, keys, attn_weights, causal_mask, seq, heads, dim):  
    """  
    Apply causal mask in ciphertext domain  
    
    Parameters:  
    cc: Crypto context  
    keys: Key pair  
    attn_weights: List of encrypted attention weights, packed by heads and columns  
                  [head1_col1, head1_col2, ..., head1_dim,   
                   head2_col1, head2_col2, ..., head2_dim, ...]  
    causal_mask: Causal mask list, each column is part of an seq x seq upper triangular matrix  
    seq: Sequence length  
    heads: Number of attention heads  
    dim: Number of columns per head  
    
    Returns:  
    masked_attn_weights: List of encrypted attention weights after applying mask  
    """  
    masked_attn_weights = []  
    
    # Iterate through each head  
    for head in range(heads):  
        # Starting index for current head  
        head_start = head * dim  
        
        # Iterate through each column in current head  
        for col in range(dim):  
            # Get current column ciphertext  
            current_col_cipher = attn_weights[head_start + col]  
            
            # Get corresponding mask column  
            mask_col = causal_mask[col]  
            
            # Create plaintext for mask  
            mask_plaintext = cc.MakeCKKSPackedPlaintext(mask_col, 1, 0, None, seq)  
            
            # Apply mask in ciphertext domain (addition)  
            masked_col_cipher = cc.EvalAdd(current_col_cipher, mask_plaintext)  
            
            masked_attn_weights.append(masked_col_cipher)  
    
    return masked_attn_weights  

# Helper function: Generate causal mask  
def generate_causal_mask(seq):  
    """  
    Generate causal mask (upper triangular matrix)  
    
    Parameters:  
    seq: Sequence length  
    
    Returns:  
    causal_mask: seq x seq upper triangular mask list  
    """  
    causal_mask = []  
    
    # Generate mask for each column  
    for col in range(seq):  
        mask_col = [0.0] * seq  
        # Set upper triangular part to zero  
        for row in range(col + 1):  
            mask_col[row] = 0.0  
        # Set lower triangular part to extremely negative value (-10000)  
        for row in range(col + 1, seq):  
            mask_col[row] = -10000.0  
        
        causal_mask.append(mask_col)  
    
    return causal_mask
