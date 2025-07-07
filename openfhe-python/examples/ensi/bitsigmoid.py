
from openfhe import *  
import time  
import numpy as np  
import random  
import math  


class SigmoidCipher:
    def __init__(self, cc, keys, seq, heads, dim, 
                 lower_bound=-10, upper_bound=11, poly_degree=16):
        """
        Initialize SigmoidCipher
        
        Parameters:
        cc: Crypto context
        keys: Key pair
        seq: Sequence length
        heads: Number of attention heads
        dim: Number of columns per head
        lower_bound: Lower bound for Sigmoid function
        upper_bound: Upper bound for Sigmoid function
        poly_degree: Degree of polynomial approximation
        """
        self.cc = cc
        self.keys = keys
        self.seq = seq
        self.heads = heads
        self.dim = dim
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.poly_degree = poly_degree
    
    def forward(self, input_cols):
        """
        Compute Sigmoid in ciphertext domain
        
        Parameters:
        input_cols: Input ciphertext list, packed by heads and columns
                   Dimension (heads * dim)
        
        Returns:
        sigmoid_cols: List of ciphertexts after Sigmoid transformation
        """
        sigmoid_cols = []
        scale_plaintext = self.cc.MakeCKKSPackedPlaintext([-math.log(self.seq)] * self.seq, 1, 0, None, self.seq) 
        
        # Iterate through each head
        for head in range(self.heads):
            # Starting index for current head
            head_start = head * self.dim

            for col in range(self.dim):
                # Get current column ciphertext
                current_col_cipher = input_cols[head_start + col]
                current_col_cipher = self.cc.EvalAdd(current_col_cipher, scale_plaintext) 
                
                # Use EvalLogistic for sigmoid approximation
                sigmoid_col_cipher = self.cc.EvalLogistic(
                    current_col_cipher, 
                    self.lower_bound, 
                    self.upper_bound, 
                    self.poly_degree
                )
                
                # # SiLU implementation
                # sigmoid_col_cipher = self.cc.EvalMult(current_col_cipher, sigmoid_col_cipher)
                
                sigmoid_cols.append(sigmoid_col_cipher)

        return sigmoid_cols
