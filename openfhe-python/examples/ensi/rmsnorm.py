
from openfhe import *  
import time  
import numpy as np  
import random  
import math  


class RMSNormCipher:
    def __init__(self, cc, keys, seq, heads, dim, gamma):
        """
        Initialize RMSNormCipher

        Parameters:
        cc: Crypto context
        keys: Key pair
        seq: Sequence length
        heads: Number of attention heads
        dim: Number of columns per head
        gamma: Scaling parameter for normalization
        """
        self.cc = cc
        self.keys = keys
        self.seq = seq
        self.heads = heads
        self.dim = dim
        self.gamma = gamma 
    
    def forward(self, input_cols):
        """
        Compute RMSNorm in ciphertext domain

        Parameters:
        input_cols: Input ciphertext list, packed by heads and columns
                    Dimension (heads * dim)

        Returns:
        normalized_cols: List of ciphertexts after RMSNorm
        """
        normalized_cols = []
        
        # Iterate through each head
        for head in range(self.heads):
            # Starting index for current head
            head_start = head * self.dim
            sum_squared_cipher = self.cc.MakeCKKSPackedPlaintext([0] * self.seq, 1, 0, None, self.seq)
            
            # Iterate through each column in current head
            for col in range(self.dim):
                # Get current column ciphertext
                current_col_cipher = input_cols[head_start + col]
                
                # Calculate square
                squared_col_cipher = self.cc.EvalMult(current_col_cipher, current_col_cipher)
                
                # Sum of squares
                sum_squared_cipher = self.cc.EvalAdd(squared_col_cipher, sum_squared_cipher)

            # Create plaintext for 1/seq
            inv_seq_plaintext = self.cc.MakeCKKSPackedPlaintext([1.0/float(self.dim)] * self.seq, 1, 0, None, self.seq)
            
            # Calculate variance (mean of squares)
            variance_cipher = self.cc.EvalMult(inv_seq_plaintext, sum_squared_cipher)
            
            # print(f"Initial number of levels remaining1: {35 - variance_cipher.GetLevel()}")
            # Calculate square root (using approximation function)
            lower_bound = 0
            upper_bound = 100
            poly_degree = 60  # 60
            sqrt_variance_cipher = self.cc.EvalChebyshevFunction(math.sqrt, variance_cipher, lower_bound, upper_bound, poly_degree)
            
            print(f"Initial number of levels remaining2: {35 - sqrt_variance_cipher.GetLevel()}")
            # Calculate reciprocal (for normalization)
            # Using division approximation function
            lower_bound = 0.1
            upper_bound = 10
            normalized_col_cipher = self.cc.EvalDivide(sqrt_variance_cipher, lower_bound, upper_bound, poly_degree)
            print(f"Initial number of levels remaining3: {35 - normalized_col_cipher.GetLevel()}")
            
            # Iterate through each column in current head
            for col in range(self.dim):
                # Get current column ciphertext
                current_col_cipher = input_cols[head_start + col]

                # Multiply normalized column with original column
                rmsnorm_col_cipher = self.cc.EvalMult(current_col_cipher, normalized_col_cipher)
                rmsnorm_col_cipher = self.cc.EvalMult(rmsnorm_col_cipher, self.gamma)
                normalized_cols.append(rmsnorm_col_cipher)
        
        return normalized_cols
