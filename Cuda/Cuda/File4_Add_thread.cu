#include<iostream>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#define N (1024)

__global__ void add(int* a, int* b, int* c)
{
	int tid = threadIdx.x;
	if (tid < N)
	{
		c[tid] = a[tid] + b[tid];
	}
	return;
}

int main(void) 
{
	int a[N], b[N], c[N];
	int* dev_a, * dev_b, * dev_c;

	cudaMalloc((void**)&dev_a, sizeof(int) * N);
	cudaMalloc((void**)&dev_b, sizeof(int) * N);
	cudaMalloc((void**)&dev_c, sizeof(int) * N);

	for (int i = 0; i < N; i++)
	{
		a[i] = i;
		b[i] = i * i;
	}

	cudaMemcpy(dev_a, a, sizeof(int) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, sizeof(int) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_c, c, sizeof(int) * N, cudaMemcpyHostToDevice);

	add <<<1, N >>> (dev_a, dev_b, dev_c);

	cudaMemcpy(c, dev_c, sizeof(int) * N, cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; i++)
	{
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
	}
	
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

}