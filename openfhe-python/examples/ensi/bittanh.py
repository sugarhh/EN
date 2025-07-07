
from openfhe import *  
import time  
import numpy as np  
import random  
import math  


class TanhCipher:  
    def __init__(self, cc, keys, seq, heads, dim,   
                 lower_bound=-10, upper_bound=11, poly_degree=16):  
        """  
        Initialize TanhCipher  

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
        Compute Tanh in ciphertext domain  

        Parameters:  
        input_cols: Input ciphertext list, packed by heads and columns  
                    Dimension (heads * dim)  

        Returns:  
        tanh_cols: List of ciphertexts after Tanh transformation  
        """  
        tanh_cols = []  
        
        # Iterate through each head  
        for head in range(self.heads):  
            # Starting index for current head  
            head_start = head * self.dim  
            
            # Iterate through each column in current head  
            for col in range(self.dim):  
                # Get current column ciphertext  
                current_col_cipher = input_cols[head_start + col]  
                
                # Calculate 2x  
                # two_plaintext = self.cc.MakeCKKSPackedPlaintext([2.0] * self.seq, 1, 0, None, self.seq)
                two_x_cipher = self.cc.EvalMult(current_col_cipher, 2)  
                
                # Use EvalLogistic for sigmoid approximation  
                sigmoid_2x_cipher = self.cc.EvalLogistic(  
                    two_x_cipher,   
                    self.lower_bound,   
                    self.upper_bound,   
                    self.poly_degree  
                )  
                
                # Calculate tanh = 2 * sigmoid(2x) - 1  
                tanh_col_cipher = self.cc.EvalAdd(self.cc.EvalMult(sigmoid_2x_cipher, 2), -1)  
                
                tanh_cols.append(tanh_col_cipher)  
        
        return tanh_cols
