# ENSI: Efficient Non-Interactive Secure Inference for Large Language Models

## Overview
ENSI is an efficient, non-interactive secure inference framework for large language models (LLMs). By jointly designing cryptographic protocols and model architecture, ENSI enables privacy-preserving inference on sensitive data without exposing user information.

- **A Co-Design Secure Inference Framework for LLM:**  
  - Proposes a framework that jointly integrates encryption schemes, encoding strategies, and model optimization for secure LLM inference.  
  - Introduces a column-wise encoding method, specifically designed to support multi-head self-attention, enabling nonlinear operations to follow matrix multiplications without the need for expensive ciphertext interleaving.  
  - Optimizes the matrix multiplication process based on the structure of BitNet, effectively removing the need for multiplicative operations within the PCMM, which significantly reduces computational complexity.

- **Retraining-Free HE Implementation for Non-Linear Functions:**  
  - Addresses the high computational overhead of softmax under homomorphic encryption by leveraging sigmoid attention as an efficient and effective drop-in replacement.  

- ** Efficient Implementation on CPU and GPU:** 

## Requirements
