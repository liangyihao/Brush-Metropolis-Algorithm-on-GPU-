# Brush-Metropolis-Algorithm-on-GPU-
Brush Metropolis Algorithm, Described in the paper [A GPU-based large-scale Monte Carlo simulation method for systems with long-range interactions](http://www.sciencedirect.com/science/article/pii/S0021999117301729)
is an efficient implementation of Canonical Monte Carlo simulation for N-body systems on graphics processing units (GPU). 
Our method takes advantage of the GPU Single Instruction, Multiple Data (SIMD) architectures, and adopts the sequential updating scheme of Metropolis algorithm. It makes no approximation in the computation of energy, and reaches a remarkable 440-fold speedup, compared with the serial implementation on CPU. 

Here we only present the main code of this method. This code is in CUDA programming language and is available on NVIDIA Tesla K20 and higher.
