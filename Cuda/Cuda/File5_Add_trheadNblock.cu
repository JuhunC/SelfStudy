#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<common/book.h>
#include<chrono>
#include<stdio.h>

#define CUDA_MAX_THREAD	(1024)
#define N				(33 * CUDA_MAX_THREAD)

auto getTime(void)
{
	return std::chrono::high_resolution_clock::now();
}

__global__ void cudaAdd(int* a, int* b, int* c)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	while (tid < N)
	{
		c[tid] = a[tid] + b[tid];
		tid += blockDim.x * gridDim.x;
	}
}
__host__ void Add(int* a, int* b, int* c)
{
	for (int i = 0; i < N; i++)
	{
		c[i] = a[i] + b[i];
	}
}
void CPU(int* a, int* b, int* c)
{
	Add(a, b, c);
}
void GPU(int* a, int* b, int* c)
{
	int* dev_a, * dev_b, * dev_c;

	cudaMalloc((void**)&dev_a, sizeof(int) * N);
	cudaMalloc((void**)&dev_b, sizeof(int) * N);
	cudaMalloc((void**)&dev_c, sizeof(int) * N);

	cudaMemcpy(dev_a, a, sizeof(int) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, sizeof(int) * N, cudaMemcpyHostToDevice);
	auto s1 = getTime();

	cudaAdd << <128, 128 >> > (dev_a, dev_b, dev_c);
	auto e1 = getTime();
	auto elapsed = (std::chrono::nanoseconds)(e1 - s1);
	printf("[%.5f seconds] GPU Add Function Runtime\n", elapsed.count() * 1e-9);

	cudaMemcpy(c, dev_c, sizeof(int) * N, cudaMemcpyDeviceToHost);

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	
	return;
}

int main(void)
{
	int a[N], b[N], c[N];

	for (int i = 0; i < N; i++)
	{
		a[i] = i;
		b[i] = i * i;
	}
	auto s1 = getTime();
	GPU(a, b, c);
	auto e1 = getTime();
	auto elapsed = (std::chrono::nanoseconds)(e1 - s1);

	printf("[%.5f seconds] GPU Runtime(including MemCpy/Alloc/Dealloc)\n", elapsed.count() * 1e-9);

	auto s2 = getTime();
	CPU(a, b, c);
	auto e2 = getTime();
	auto elapsed1 = (std::chrono::nanoseconds)(e2 - s2);

	printf("[%.5f seconds] CPU Runtime\n", elapsed1.count() * 1e-9);

	return;
}