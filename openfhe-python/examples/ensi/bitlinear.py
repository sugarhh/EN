
from openfhe import *  
import time  
import numpy as np   
import random
from tqdm import tqdm

def matrix_mult_plain(cc, keys, enc_matrix, plain_matrix, seq, input_dim, output_dim):  
    """  
    Compute multiplication between encrypted matrix and plaintext matrix
    
    Parameters:  
    cc: Crypto context
    keys: Key pair
    enc_matrix: List containing encrypted matrix columns [col0_enc, col1_enc, ...]  
    plain_matrix: 2D list, plaintext matrix with dimension input_dim x output_dim (contains only 0, 1, -1)
    seq: Number of rows in encrypted matrix
    input_dim: Number of columns in input matrix (also number of rows in plain_matrix)
    output_dim: Number of columns in output matrix (also number of columns in plain_matrix)
    
    Returns:  
    result_cols: List containing encrypted columns of the result matrix
    """  
    print("\n=== Performing encrypted matrix multiplication with plaintext matrix ===")  
    result_cols = []  
    zero_vector = [0.0] * seq  
    
    # Calculate for each output column
    for j in tqdm(range(output_dim)):
        # Check if current column is all zeros
        is_zero_column = all(plain_matrix[i][j] == 0 for i in range(input_dim))
        
        if is_zero_column:
            # If all zeros, directly use original plaintext encryption
            zero_plaintext_new = cc.MakeCKKSPackedPlaintext(zero_vector, 1, 0, None, seq)
            result_cols.append(cc.Encrypt(keys.publicKey, zero_plaintext_new))
            continue
            
        result_col = None
        first_add = True
        
        # Process each input column
        for i in range(input_dim):
            # Get element at position (i,j) from plain matrix
            multiplier = plain_matrix[i][j]
            
            if multiplier == 0:
                continue
            elif multiplier == 1:
                if first_add:
                    result_col = enc_matrix[i]
                    first_add = False
                else:
                    result_col = cc.EvalAdd(result_col, enc_matrix[i])
            elif multiplier == -1:
                neg_col = cc.EvalNegate(enc_matrix[i])
                if first_add:
                    result_col = neg_col
                    first_add = False
                else:
                    result_col = cc.EvalAdd(result_col, neg_col)
        
        # If no operations were performed for this column, use a new zero vector
        if first_add:
            zero_plaintext_new = cc.MakeCKKSPackedPlaintext(zero_vector, 1, 0, None, seq)
            result_col = cc.Encrypt(keys.publicKey, zero_plaintext_new)
            
        # Store encrypted result column
        result_cols.append(result_col)
    
    return result_cols


def weight_quant_cipher_mult(cc, keys, enc_matrix, weight_matrix, scale, seq, input_dim, output_dim):
    """  
    Encrypted weight quantization and matrix multiplication
    
    Parameters:
    cc: Crypto context
    keys: Key pair
    enc_matrix: Encrypted input matrix (seq x input_dim)
    weight_matrix: Plaintext weight matrix (input_dim x output_dim)
    scale: Plaintext scale value
    seq: Number of rows in encrypted matrix
    input_dim: Input dimension
    output_dim: Output dimension
    
    Returns:
    result_cols: List containing encrypted columns of the final result matrix
    """
    # Step 1: Perform multiplication between encrypted matrix and plaintext weight matrix
    intermediate_result = matrix_mult_plain(cc, keys, enc_matrix, weight_matrix, seq, input_dim, output_dim)
    
    # Step 2: Multiply the result with scale
    final_result = []
    
    # Create plaintext vector for scale
    scale_plaintext = cc.MakeCKKSPackedPlaintext([scale] * seq, 1, 0, None, seq)
    
    # Apply scale to each column
    for j in range(output_dim):
        if intermediate_result[j] is not None:
            final_result.append(cc.EvalMult(intermediate_result[j], scale_plaintext))
    
    return final_result


class CipherBitLinear:
    def __init__(self, cc, keys, weight_matrix, scale, seq, input_dim, output_dim):
        """
        Initialize CipherBitLinear
        
        Parameters:
        cc: Crypto context
        keys: Key pair
        weight_matrix: Plaintext weight matrix
        scale: Quantization scale value
        seq: Sequence length
        input_dim: Input dimension
        output_dim: Output dimension
        """
        self.cc = cc
        self.keys = keys
        self.weight_matrix = weight_matrix  # Store the weight matrix directly
        self.scale = scale
        self.seq = seq
        self.input_dim = input_dim
        self.output_dim = output_dim
        
    def forward(self, enc_input):
        """
        Forward propagation
        
        Parameters:
        enc_input: Encrypted input matrix (in list form, containing columns' ciphertexts)
        
        Returns:
        Encrypted output matrix
        """
        # Perform weight quantization and matrix multiplication
        return weight_quant_cipher_mult(
            self.cc,
            self.keys,
            enc_input,
            self.weight_matrix,
            self.scale,
            self.seq,
            self.input_dim,
            self.output_dim
        )
