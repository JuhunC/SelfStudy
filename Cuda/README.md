# Cuda

- Function, Variables
```
__HOST__   : Runs on CPU, called from CPU
__GLOBAL__ : Runs on GPU, called from CPU, GPU
__DEVICE__ : Runs on GPU, called from GPU
```

- Two - dimensional indexing
```
int tid = threadIdx.x + blockIdx.x * blockDim.x;
/// Index = (Thread Offset) + (Block Offset * Block Size)
```