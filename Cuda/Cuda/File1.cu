#include<iostream>
#include<cuda_runtime.h>

int main(void)
{
	cudaDeviceProp prop;

	int count;
	cudaGetDeviceCount(&count);

	for (int i = 0; i < count; i++)
	{
		cudaGetDeviceProperties(&prop, i);

	}



	return 0;
}