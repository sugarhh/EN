
from openfhe import *  
import time  
import numpy as np   
from bitlinear import CipherBitLinear
from bitsigmoid import SigmoidCipher
from rmsnorm import RMSNormCipher
import random


class CipherBitnetMLP:  
    def __init__(self, cc, keys, seq, heads, dim, intermediate_size,  
                 gate_proj_weights,   
                 up_proj_weights,   
                 down_proj_weights):  
        """  
        Initialize Bitnet MLP layer in ciphertext domain
        
        Parameters:  
        cc: Crypto context  
        keys: Key pair  
        seq: Sequence length  
        heads: Number of attention heads   
        dim: Dimension size  
        intermediate_size: Size of intermediate layer  
        gate_proj_weights: Weight matrix for gate_proj  
        up_proj_weights: Weight matrix for up_proj  
        down_proj_weights: Weight matrix for down_proj  
        """  
        self.cc = cc  
        self.keys = keys   
        self.seq = seq  
        self.heads = heads  
        self.dim = dim  
        self.intermediate_size = intermediate_size  
        
        # Use provided weight matrices  
        self.gate_proj = CipherBitLinear(  
            cc, keys,   
            weight_matrix=gate_proj_weights,  
            scale=1.0,  
            seq=self.seq,  
            input_dim=self.heads * self.dim,   
            output_dim=self.intermediate_size  
        )  
        
        self.up_proj = CipherBitLinear(  
            cc, keys,   
            weight_matrix=up_proj_weights,  
            scale=1.0,  
            seq=self.seq,  
            input_dim=self.heads * self.dim,   
            output_dim=self.intermediate_size  
        )  
        
        self.down_proj = CipherBitLinear(  
            cc, keys,   
            weight_matrix=down_proj_weights,  
            scale=1.0,  
            seq=self.seq,  
            input_dim=self.intermediate_size,   
            output_dim=self.heads * self.dim  
        )  
        
        # Initialize Sigmoid and RMSNorm  
        self.sigmoid = SigmoidCipher(  
            cc, keys,   
            seq=self.seq,   
            heads=self.heads,   
            dim=self.intermediate_size // self.heads
        )  
        
        self.ffn_layernorm = RMSNormCipher(  
            cc, keys,   
            seq=self.seq,   
            heads=self.heads,   
            dim=self.intermediate_size //self.heads  
        )    

    def silu_activation(self, gate_cols):  
        """  
        Implement SiLU (Sigmoid Linear Unit) activation function  
        SiLU(x) = x * sigmoid(x)  
        
        Parameters:  
        gate_cols: Input columns for Sigmoid  
        
        Returns:  
        Columns after SiLU activation  
        """  
        # Calculate sigmoid(gate_cols)  
        sigmoid_gate = self.sigmoid.forward(gate_cols)  
        
        # Calculate x * sigmoid(x) for each column  
        silu_output = []  
        for i in range(len(gate_cols)):  
            # gate_cols[i] as original input  
            # sigmoid_gate[i] as sigmoid part  
            silu_output.append(  
                self.cc.EvalMult(gate_cols[i], sigmoid_gate[i])  
            )  
        
        return silu_output  

    def forward(self, input_cols):  
        """  
        Forward propagation  
        
        Parameters:  
        input_cols: List of encrypted inputs  
        
        Returns:  
        List of final encrypted outputs  
        """  
        # gate_proj operation  
        gate_output = self.gate_proj.forward(input_cols)  
        
        # up_proj operation  
        up_output = self.up_proj.forward(input_cols)  
        
        # SiLU activation: x * sigmoid(x)  
        silu_output = self.silu_activation(gate_output)  
        
        # Use up_proj output for Hadamard product with SiLU  
        hadamard_output = []  
        for i in range(len(silu_output)):  
            hadamard_output.append(  
                self.cc.EvalMult(silu_output[i], up_output[i])  
            )  
        
        # RMSNorm  
        # normalized_output = self.ffn_layernorm.forward(hadamard_output)  
        normalized_output = self.sigmoid.forward(hadamard_output)
        
        # # # down_proj operation  
        final_output = self.down_proj.forward(normalized_output)  
        
        return final_output
