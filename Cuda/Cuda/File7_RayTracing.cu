#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <common/cpu_bitmap.h>
#include <common/book.h>
#include<iostream>

#define INF 2e10f
#define DIM (100)
struct Sphere
{
	float r, g, b;
	float radius;
	float x, y, z;
	__device__ float hit(float ox, float oy, float* n)
	{
		float dx = ox - x;
		float dy = oy - y;
		
		if (dx * dx + dy * dy < radius * radius)
		{
			float dz = sqrtf(radius * radius - dx * dx - dy * dy);
			*n = dz / sqrtf(radius * radius);
			return dz + z;
		}
		return -INF;
	}
};

#define rnd(x) (x*rand()/RAND_MAX)
#define SPHERES 5

Sphere* s;

__device__ void kernel(unsigned char* ptr)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	int offset = x + y * blockDim.x * gridDim.x;
	float ox = (x - DIM / 2);
	float oy = (y - DIM / 2);

	float r = 0, g = 0, b = 0;
	float maxz = -INF;

	for (int i = 0; i < SPHERES; i++)
	{
		float n;
		float t = s[i].hit(ox, oy, &n);
		if (t > maxz)
		{
			float fscale = n;
			r = s[i].r * fscale;
			g = s[i].g * fscale;
			b = s[i].b * fscale;
		}

		ptr[offset * 4 + 0] = (int)(r * 255);
		ptr[offset * 4 + 1] = (int)(g * 255);
		ptr[offset * 4 + 2] = (int)(b * 255);
		ptr[offset * 4 + 3] = 255;
	}

}

int main(void)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	CPUBitmap bitmap(DIM, DIM);
	
	unsigned char* dev_bitmap;

	cudaMalloc((void**)&dev_bitmap, bitmap.image_size());
	cudaMalloc((void**)&s, sizeof(Sphere) * SPHERES);

	Sphere* tmp_s = (Sphere*)malloc(sizeof(Sphere) * SPHERES);

	for (int i = 0; i < SPHERES; i++)
	{
		tmp_s[i].r = rnd(1.0f);
		tmp_s[i].g = rnd(1.0f);
		tmp_s[i].b = rnd(1.0f);
		tmp_s[i].x = rnd(1000.0f) - 500;
		tmp_s[i].y = rnd(1000.0f) - 500;
		tmp_s[i].z = rnd(1000.0f) - 500;
		tmp_s[i].radius = rnd(100.0f) + 20;
	}

	cudaMemcpy(dev_bitmap, tmp_s, sizeof(Sphere) * SPHERES, cudaMemcpyHostToDevice);

	free(tmp_s);

	dim3 grids(DIM / 16, DIM / 16);
	dim3 threads(16, 16);
	kernel <<<grids, threads >>> (dev_bitmap);

	cudaMemcpy(bitmap.get_bitmap_ptr(), dev_bitmap, sizeof(Sphere) * SPHERES, cudaMemcpyDeviceToHost);

	bitmap.display_and_exit();

	cudaFree(dev_bitmap);
	cudaFree(s);

	return 0;
}