
from openfhe import *  
import time  
import numpy as np   
import math
import random


def generate_universal_vectors_homomorphic_bsgs(cc, keys, enc_sequences, n, m):
    """  
    Generate universal vectors for encrypted input sequences using homomorphic operations
    Uses BSGS strategy to reduce rotation count from O(m) to O(log m)
    
    Args:  
        cc: Crypto context
        keys: Key pair
        enc_sequences: List of encrypted sequences, each is an encrypted vector
        n: Number of ciphertexts (sequence length)
        m: Length of each ciphertext vector
        
    Returns:  
        List containing all generated encrypted repeated vectors
    """  
    
    results = []  
    
    # Generate masks
    masks = []  
    for j in range(m):  
        mask_vector = [1 if idx == j else 0 for idx in range(m)]  
        mask_ptxt = cc.MakeCKKSPackedPlaintext(mask_vector, 1, 0, None, m)  
        masks.append(mask_ptxt)  

    steps = []  
    step = m // 2  
    while step >= 1:  
        steps.append(step)  
        step = step // 2 
    # print(steps) 
    
    # Process each encrypted sequence
    for enc_sequence in enc_sequences:  
        # Generate repeated vectors for each position (copy each position's element to all slots)
        for i in range(m):  
            # Step 1: Apply mask to extract target element
            # print("dididi",i)
            masked = cc.EvalMult(enc_sequence, masks[i])  
            
            current = masked
            for k in range(len(steps)):
                # print(steps[k])
                rotated = cc.EvalRotate(current, steps[k])
                current = cc.EvalAdd(current, rotated)
                  
            # Add directly to results list without intermediate variables
            results.append(current)  
    
    return results  


def generate_universal_vectors_homomorphic(cc, keys, enc_sequences, n, m):
    """  
    Generate universal vectors for encrypted input sequences using homomorphic operations
    
    Args:  
        cc: Crypto context
        keys: Key pair
        enc_sequences: List of encrypted sequences, each is an encrypted vector
        n: Number of ciphertexts (sequence length)
        m: Length of each ciphertext vector
        
    Returns:  
        List containing all generated encrypted repeated vectors
    """  
    
    results = []  
    masks = []  
    for j in range(m):  
        mask_vector = [1 if idx == j else 0 for idx in range(m)]  
        mask_ptxt = cc.MakeCKKSPackedPlaintext(mask_vector, 1, 0, None, m)  
        masks.append(mask_ptxt)  
    
    # Process each encrypted sequence
    for enc_sequence in enc_sequences:  
        # Generate all rotated versions
        rotations = []  
        for k in range(m):  
            rotated = cc.EvalRotate(enc_sequence, k)  
            rotations.append(rotated)  
        
        # Generate repeated vector for each position
        for i in range(m):  
            target = None  
            first_add = True  
            
            # Apply rotation and mask for each position
            for j in range(m):  
                k = (i - j) % m  # Calculate required rotation version
                rotated = rotations[k]  
                # print(k)
                # Apply mask (multiply with mask)
                masked_rotated = cc.EvalMult(rotated, masks[j])  
                
                # Accumulate results
                if first_add:  
                    target = masked_rotated  
                    first_add = False  
                else:  
                    target = cc.EvalAdd(target, masked_rotated)  
            
            results.append(target)  
    
    return results  


def matrix_multiplication_homomorphic(cc, keys, matrix_A_cols, processed_B, n, p):
    """  
    Perform homomorphic matrix multiplication A × B
    
    Args:  
        cc: Crypto context
        keys: Key pair
        matrix_A_cols: List of column vectors for matrix A (encrypted)
        processed_B: Preprocessed elements of matrix B (result of generate_universal_vectors_homomorphic)
        n: Number of columns in matrix A (also rows in B)
        p: Number of columns in matrix B
        
    Returns:  
        List of column vectors for result matrix C (encrypted)
    """  
    result_cols = []  
    
    # Calculate each column of the result matrix
    for j in range(p):  
        # Initialize result column vector
        result_col = None  
        first_add = True  
        
        # Calculate contribution of each column of A with corresponding row of B
        for k in range(n):  
            # Get column k of A
            col_A = matrix_A_cols[k]  
            
            # Get element at row k, column j of B (already preprocessed into vector form)
            # Index calculation in processed_B: k*p + j represents element at row k, column j
            element_B = processed_B[k*p + j]  
            
            # Calculate contribution of current term: column vector of A * corresponding element of B
            contribution = cc.EvalMult(col_A, element_B)  
            
            # Accumulate to result column
            if first_add:  
                result_col = contribution  
                first_add = False  
            else:  
                result_col = cc.EvalAdd(result_col, contribution)  
        
        result_cols.append(result_col)  
    
    return result_cols   


def group_cipher_matrix_mult(cc, keys, matrix_A, matrix_B, seq, heads, dim):
    """  
    Grouped computation of ciphertext matrix multiplication
    
    Parameters:  
    cc: Crypto context
    keys: Key pair
    matrix_A: List containing columns of encrypted matrix A [col0_enc, col1_enc, ...]
             Dimension (heads, seq, dim), stored by head groups, each group has dim columns, each column has length seq
    matrix_B: List containing rows of encrypted matrix B [row0_enc, row1_enc, ...]
             Dimension (heads, dim, seq), stored by head groups, each group has dim rows, each row has length seq
    seq: Sequence length
    heads: Number of attention heads
    dim: Dimension of each head
    
    Returns:  
    result_cols: List containing columns of result matrix [col0_enc, col1_enc, ...]
                Each group stores seq columns consecutively, with heads groups in total
    """  
    result_cols = []  
    
    # Perform separate matrix multiplication for each head
    for h in range(heads):  
        # Get matrices for current head
        current_A_cols = []  
        current_B_rows = []  
        
        # Extract columns of matrix A for current head
        base_idx_A = h * dim  
        for k in range(dim):  
            current_A_cols.append(matrix_A[base_idx_A + k])  
            
        # Extract rows of matrix B for current head
        base_idx_B = h * dim  
        for k in range(dim):  
            current_B_rows.append(matrix_B[base_idx_B + k])  
            
        # # # Preprocess matrix B
        # processed_B = generate_universal_vectors_homomorphic_bsgs(  
        #     cc,   
        #     keys,  
        #     current_B_rows,  
        #     dim,  
        #     seq  
        # )

        processed_B = generate_universal_vectors_homomorphic(  
            cc,   
            keys,  
            current_B_rows,  
            dim,  
            seq  
        )  
        
        # Perform matrix multiplication
        head_result_cols = matrix_multiplication_homomorphic(  
            cc,  
            keys,  
            current_A_cols,  
            processed_B,    
            dim,  
            seq  
        )  
        
        # Store results for current head in consecutive positions
        result_cols.extend(head_result_cols)  
    
    return result_cols  


class CipherMatrixMult:  
    def __init__(self, cc, keys, heads, dim, seq):  
        """  
        Initialize CipherMatrixMult
        
        Parameters:  
        cc: Crypto context
        keys: Key pair
        heads: Number of attention heads
        dim: Dimension of each head
        seq: Sequence length
        """  
        self.cc = cc  
        self.keys = keys  
        self.heads = heads  
        self.dim = dim  
        self.seq = seq  
        
    def forward(self, matrix_A, matrix_B):  
        """  
        Forward propagation
        
        Parameters:  
        matrix_A: Encrypted input matrix A (list format, containing columns' ciphertexts)
                Dimension (heads, seq, dim)
        matrix_B: Encrypted input matrix B (list format, containing rows' ciphertexts)
                Dimension (heads, dim, seq)
        
        Returns:  
        mult_result: List containing columns of result matrix [col0_enc, col1_enc, ...]
                    Each group stores seq columns consecutively, with heads groups in total
        """  
        # Perform grouped matrix multiplication
        mult_result = group_cipher_matrix_mult(  
            self.cc,  
            self.keys,  
            matrix_A,  
            matrix_B,  
            self.seq,  
            self.heads,  
            self.dim  
        )  

        # # Perform scaling
        # scale = 1.0 / math.sqrt(self.dim)  
        # scale_plaintext = self.cc.MakeCKKSPackedPlaintext([scale], 1, 0, None, self.seq)  

        # # Scale all columns
        # scaled_cols = []  
        # for i in range(self.heads * self.seq):  
        #     if mult_result[i] is not None:  
        #         scaled_cols.append(self.cc.EvalMult(mult_result[i], scale_plaintext))  

        return mult_result
