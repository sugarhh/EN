# ENSI: Efficient Non-Interactive Secure Inference for Large Language Models

## Overview
ENSI is an efficient, non-interactive secure inference framework for large language models (LLMs). By jointly designing cryptographic protocols and model architecture, ENSI enables privacy-preserving inference on sensitive data without exposing user information.

- **A Co-Design Secure Inference Framework for LLM:**  
  - Proposes a framework that jointly integrates encryption schemes, encoding strategies, and model optimization for secure LLM inference.  
  - Introduces a column-wise encoding method, specifically designed to support multi-head self-attention, enabling nonlinear operations to follow matrix multiplications without the need for expensive ciphertext interleaving.  
  - Optimizes the matrix multiplication process based on the structure of BitNet, effectively removing the need for multiplicative operations within the PCMM, which significantly reduces computational complexity.

- **Retraining-Free HE Implementation for Non-Linear Functions:**  
  - Addresses the high computational overhead of softmax under homomorphic encryption by leveraging sigmoid attention as an efficient and effective drop-in replacement.  

- **Efficient Implementation on CPU and GPU:** 

## Dependencies
- [OpenFHE](https://github.com/openfheorg/openfhe-python): Used as our main homomorphic encryption backend, providing efficient CKKS implementation and cryptographic primitives.
- [Microsoft SEAL](https://github.com/microsoft/SEAL): Used for compatibility testing and cross-library benchmarks.
- [Phantom](https://github.com/MrSlavika/phantom-fhe-boot): A CUDA-Accelerated Fully Homomorphic Encryption Library.
- [1bitLLM/bitnet_b1_58-large](https://huggingface.co/1bitLLM/bitnet_b1_58-large): The open-source implementation and pre-trained weights for BitNet, integrated as the backbone LLM in our framework.

## Directory Structure
- `openfhe-python/examples/ensi` - Main source code

