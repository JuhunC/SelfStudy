# Cuda

Cuda Programming Model: https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/


## Function, Variables
```__HOST__```   : Runs on CPU, called from CPU

```__GLOBAL__``` : Runs on GPU, called from CPU, GPU

```__DEVICE__``` : Runs on GPU, called from GPU

## Two - dimensional indexing
```
int tid = threadIdx.x + blockIdx.x * blockDim.x;
/// Index = (Thread Offset) + (Block Offset * Block Size)
```

## GPU Memory
### Shared Memory
- Shared Mememory can be used using ```__shared__``` keyword
- Variables in shared memory is creates a copy for each block.
- These variables cannot be accessed from other blocks
- **WARNING:Problem with Racing Condition Exists**
- *Synchronizing threads by ***__syncthreads()***;*


# References
- https://www.nvidia.com/docs/IO/116711/sc11-cuda-c-basics.pdf
