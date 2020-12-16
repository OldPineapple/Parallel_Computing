
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "gputimer.h"

#define BLOCK_SIZE 32
#define GRID_SIZE 4
#define THREAD_SIZE 4
float u1[512][512];
float u2[512][512];
float u[512][512];

template<typename T>
struct array2D
{
	T* p;
	int lda;

	__device__ __host__
		array2D(T* _p, int cols) : p(_p), lda(cols) {}

	__device__ __host__
		T& operator()(int i, int j) { return p[i * lda + j]; }

	__device__ __host__
		T& operator()(int i, int j) const { return p[i * lda + j]; }
};


__global__ void calculation(array2D<float> u1, array2D<float> u2, array2D<float> u, int N, float eta, float rho, float G) {
	// Calculate the row index of the P element and M
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	// Calculate the column index of P and N
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (0 < i && i * THREAD_SIZE + THREAD_SIZE - 1 < N - 1 && 0 < j && j * THREAD_SIZE + THREAD_SIZE - 1 < N - 1)
	{
		for (int m = 0; m < THREAD_SIZE; m++)
		{
			for (int n = 0; n < THREAD_SIZE; n++)
			{
				u(i * THREAD_SIZE + m, j * THREAD_SIZE + n) = (rho * (u1(i * THREAD_SIZE - 1 + m, j * THREAD_SIZE + n)
					+ u1(i * THREAD_SIZE + 1 + m, j * THREAD_SIZE + n)
					+ u1(i * THREAD_SIZE + m, j * THREAD_SIZE - 1 + n)
					+ u1(i * THREAD_SIZE + m, j * THREAD_SIZE + 1 + n)
					- 4 * u1(i * THREAD_SIZE + m, j * THREAD_SIZE + n))
					+ 2 * u1(i * THREAD_SIZE + m, j * THREAD_SIZE + n)
					- (1 - eta) * u2(i * THREAD_SIZE + m, j * THREAD_SIZE + n)) / (1 + eta);
			}

		}
	}

	__syncthreads();

	// Boundary right thread
	if (i * THREAD_SIZE + THREAD_SIZE - 1 == N - 1 && j != 0 && j * THREAD_SIZE + THREAD_SIZE - 1 != N - 1)
	{
		for (int m = 0; m < THREAD_SIZE - 1; m++)
		{
			for (int n = 0; n < THREAD_SIZE; n++)
			{
				u(i * THREAD_SIZE + m, j * THREAD_SIZE + n) = (rho * (u1(i * THREAD_SIZE - 1 + m, j * THREAD_SIZE + n)
					+ u1(i * THREAD_SIZE + 1 + m, j * THREAD_SIZE + n)
					+ u1(i * THREAD_SIZE + m, j * THREAD_SIZE - 1 + n)
					+ u1(i * THREAD_SIZE + m, j * THREAD_SIZE + 1 + n)
					- 4 * u1(i * THREAD_SIZE + m, j * THREAD_SIZE + n))
					+ 2 * u1(i * THREAD_SIZE + m, j * THREAD_SIZE + n)
					- (1 - eta) * u2(i * THREAD_SIZE + m, j * THREAD_SIZE + n)) / (1 + eta);
			}
		}
		for (int n = 0; n < THREAD_SIZE; n++)
			u(N - 1, j * THREAD_SIZE + n) = G * u(N - 2, j * THREAD_SIZE + n);
	}
	// Boundary left thread
	else if (i == 0 && j != 0 && j * THREAD_SIZE + THREAD_SIZE - 1 != N - 1)
	{
		for (int m = 0; m < THREAD_SIZE - 1; m++)
		{
			for (int n = 0; n < THREAD_SIZE; n++)
			{
				u(i * THREAD_SIZE + THREAD_SIZE - m, j * THREAD_SIZE + THREAD_SIZE - n) = (rho * (u1(i * THREAD_SIZE - 1 + THREAD_SIZE - m, j * THREAD_SIZE + THREAD_SIZE - n)
					+ u1(i * THREAD_SIZE + 1 + THREAD_SIZE - m, j * THREAD_SIZE + THREAD_SIZE - n)
					+ u1(i * THREAD_SIZE + THREAD_SIZE - m, j * THREAD_SIZE - 1 + THREAD_SIZE - n)
					+ u1(i * THREAD_SIZE + THREAD_SIZE - m, j * THREAD_SIZE + 1 + THREAD_SIZE - n)
					- 4 * u1(i * THREAD_SIZE + THREAD_SIZE - m, j * THREAD_SIZE + THREAD_SIZE - n))
					+ 2 * u1(i * THREAD_SIZE + THREAD_SIZE - m, j * THREAD_SIZE + THREAD_SIZE - n)
					- (1 - eta) * u2(i * THREAD_SIZE + THREAD_SIZE - m, j * THREAD_SIZE + THREAD_SIZE - n)) / (1 + eta);
			}
		}
		for (int n = 0; n < THREAD_SIZE; n++)
			u(0, j * THREAD_SIZE + n) = G * u(1, j * THREAD_SIZE + n);
	}
	// Boundary Top thread
	else if (i != 0 && i * THREAD_SIZE + THREAD_SIZE - 1 != N - 1 && j == 0)
	{
		for (int m = 0; m < THREAD_SIZE; m++)
		{
			for (int n = 0; n < THREAD_SIZE - 1; n++)
			{
				u(i * THREAD_SIZE + THREAD_SIZE - m, j * THREAD_SIZE + THREAD_SIZE - n) = (rho * (u1(i * THREAD_SIZE - 1 + THREAD_SIZE - m, j * THREAD_SIZE + THREAD_SIZE - n)
					+ u1(i * THREAD_SIZE + 1 + THREAD_SIZE - m, j * THREAD_SIZE + THREAD_SIZE - n)
					+ u1(i * THREAD_SIZE + THREAD_SIZE - m, j * THREAD_SIZE - 1 + THREAD_SIZE - n)
					+ u1(i * THREAD_SIZE + THREAD_SIZE - m, j * THREAD_SIZE + 1 + THREAD_SIZE - n)
					- 4 * u1(i * THREAD_SIZE + THREAD_SIZE - m, j * THREAD_SIZE + THREAD_SIZE - n))
					+ 2 * u1(i * THREAD_SIZE + THREAD_SIZE - m, j * THREAD_SIZE + THREAD_SIZE - n)
					- (1 - eta) * u2(i * THREAD_SIZE + THREAD_SIZE - m, j * THREAD_SIZE + THREAD_SIZE - n)) / (1 + eta);
			}
		}
		for (int m = 0; m < THREAD_SIZE; m++)
			u(i * THREAD_SIZE + m, 0) = G * u(i * THREAD_SIZE + m, 1);
	}
	// Boundary Bottom thread
	else if (i != 0 && i * THREAD_SIZE + THREAD_SIZE - 1 != N - 1 && j * THREAD_SIZE + THREAD_SIZE - 1 == N - 1)
	{
		for (int m = 0; m < THREAD_SIZE; m++)
		{
			for (int n = 0; n < THREAD_SIZE - 1; n++)
			{
				u(i * THREAD_SIZE + m, j * THREAD_SIZE + n) = (rho * (u1(i * THREAD_SIZE - 1 + m, j * THREAD_SIZE + n)
					+ u1(i * THREAD_SIZE + 1 + m, j * THREAD_SIZE + n)
					+ u1(i * THREAD_SIZE + m, j * THREAD_SIZE - 1 + n)
					+ u1(i * THREAD_SIZE + m, j * THREAD_SIZE + 1 + n)
					- 4 * u1(i * THREAD_SIZE + m, j * THREAD_SIZE + n))
					+ 2 * u1(i * THREAD_SIZE + m, j * THREAD_SIZE + n)
					- (1 - eta) * u2(i * THREAD_SIZE + m, j * THREAD_SIZE + n)) / (1 + eta);
			}
		}
		for (int m = 0; m < THREAD_SIZE; m++) {
			u(i * THREAD_SIZE + m, N - 1) = G * u(i * THREAD_SIZE + m, N - 2);
		}
	}
	__syncthreads();

	// Corner Top-left thread
	if (i == 0 && j == 0)
	{
		for (int m = 0; m < THREAD_SIZE - 1; m++)
		{
			for (int n = 0; n < THREAD_SIZE - 1; n++)
			{
				u(i * THREAD_SIZE + THREAD_SIZE - m, j * THREAD_SIZE + THREAD_SIZE - n) = (rho * (u1(i * THREAD_SIZE - 1 + THREAD_SIZE - m, j * THREAD_SIZE + THREAD_SIZE - n)
					+ u1(i * THREAD_SIZE + 1 + THREAD_SIZE - m, j * THREAD_SIZE + THREAD_SIZE - n)
					+ u1(i * THREAD_SIZE + THREAD_SIZE - m, j * THREAD_SIZE - 1 + THREAD_SIZE - n)
					+ u1(i * THREAD_SIZE + THREAD_SIZE - m, j * THREAD_SIZE + 1 + THREAD_SIZE - n)
					- 4 * u1(i * THREAD_SIZE + THREAD_SIZE - m, j * THREAD_SIZE + THREAD_SIZE - n))
					+ 2 * u1(i * THREAD_SIZE + THREAD_SIZE - m, j * THREAD_SIZE + THREAD_SIZE - n)
					- (1 - eta) * u2(i * THREAD_SIZE + THREAD_SIZE - m, j * THREAD_SIZE + THREAD_SIZE - n)) / (1 + eta);
			}
		}
		for (int m = 0; m < THREAD_SIZE - 1; m++)
		{
			u(0, j * THREAD_SIZE + m) = G * u(1, j * THREAD_SIZE + m);
			u(i * THREAD_SIZE + m, 0) = G * u(i * THREAD_SIZE + m, 1);
		}
		u(0, 0) = G * u(1, 0);
	}

	// Corner Bottom-left thread
	else if (j == 0 && i * THREAD_SIZE + THREAD_SIZE - 1 == N - 1)
	{
		for (int m = 0; m < THREAD_SIZE - 1; m++)
		{
			for (int n = 0; n < THREAD_SIZE - 1; n++)
			{
				u(i * THREAD_SIZE + m, j * THREAD_SIZE + THREAD_SIZE - n) = (rho * (u1(i * THREAD_SIZE - 1 + m, j * THREAD_SIZE + THREAD_SIZE - n)
					+ u1(i * THREAD_SIZE + 1 + m, j * THREAD_SIZE + THREAD_SIZE - n)
					+ u1(i * THREAD_SIZE + m, j * THREAD_SIZE - 1 + THREAD_SIZE - n)
					+ u1(i * THREAD_SIZE + m, j * THREAD_SIZE + 1 + THREAD_SIZE - n)
					- 4 * u1(i * THREAD_SIZE + m, j * THREAD_SIZE + THREAD_SIZE - n))
					+ 2 * u1(i * THREAD_SIZE + m, j * THREAD_SIZE + THREAD_SIZE - n)
					- (1 - eta) * u2(i * THREAD_SIZE + m, j * THREAD_SIZE + THREAD_SIZE - n)) / (1 + eta);
			}
		}
		for (int m = 0; m < THREAD_SIZE - 1; m++)
		{
			u(N - 1, THREAD_SIZE - m) = u(N - 2, THREAD_SIZE - m);
			u(i * THREAD_SIZE + m, 0) = G * u(i * THREAD_SIZE + m, 1);
		}
		u(N - 1, 0) = G * u(N - 2, 0);
	}
	// Corner Top-Right thread
	else if (i == 0 && j * THREAD_SIZE + THREAD_SIZE - 1 == N - 1)
	{
		for (int m = 0; m < THREAD_SIZE - 1; m++)
		{
			for (int n = 0; n < THREAD_SIZE - 1; n++)
			{
				u(i * THREAD_SIZE + THREAD_SIZE - m, j * THREAD_SIZE + n) = (rho * (u1(i * THREAD_SIZE - 1 + THREAD_SIZE - m, j * THREAD_SIZE + n)
					+ u1(i * THREAD_SIZE + 1 + THREAD_SIZE - m, j * THREAD_SIZE + n)
					+ u1(i * THREAD_SIZE + THREAD_SIZE - m, j * THREAD_SIZE - 1 + n)
					+ u1(i * THREAD_SIZE + THREAD_SIZE - m, j * THREAD_SIZE + 1 + n)
					- 4 * u1(i * THREAD_SIZE + THREAD_SIZE - m, j * THREAD_SIZE + n))
					+ 2 * u1(i * THREAD_SIZE + THREAD_SIZE - m, j * THREAD_SIZE + n)
					- (1 - eta) * u2(i * THREAD_SIZE + THREAD_SIZE - m, j * THREAD_SIZE + n)) / (1 + eta);
			}
		}
		for (int m = 0; m < THREAD_SIZE - 1; m++)
		{
			u(THREAD_SIZE - m, N - 1) = G * u(THREAD_SIZE - m, N - 2);
			u(0, j * THREAD_SIZE + m) = G * u(1, j * THREAD_SIZE + m);
		}
		u(0, N - 1) = G * u(0, N - 2);
	}
	// Corner Bottom-right thread
	else if (i * THREAD_SIZE + THREAD_SIZE - 1 == N - 1 && j * THREAD_SIZE + THREAD_SIZE - 1 == N - 1)
	{
		//printf("Here i*THREAD_SIZE+THREAD_SIZE - 1 is %d and j*THREAD_SIZE+THREAD_SIZE - 1 is %d\n", i*THREAD_SIZE+THREAD_SIZE - 1, j*THREAD_SIZE+THREAD_SIZE - 1);
		for (int m = 0; m < THREAD_SIZE - 1; m++)
		{
			for (int n = 0; n < THREAD_SIZE - 1; n++)
			{
				u(i * THREAD_SIZE + m, j * THREAD_SIZE + n) = (rho * (u1(i * THREAD_SIZE - 1 + m, j * THREAD_SIZE + n)
					+ u1(i * THREAD_SIZE + 1 + m, j + n)
					+ u1(i * THREAD_SIZE + m, j - 1 + n)
					+ u1(i * THREAD_SIZE + m, j + 1 + n)
					- 4 * u1(i * THREAD_SIZE + m, j + n))
					+ 2 * u1(i * THREAD_SIZE + m, j + n)
					- (1 - eta) * u2(i * THREAD_SIZE + m, j * THREAD_SIZE + n)) / (1 + eta);
			}
		}
		for (int m = 0; m < THREAD_SIZE - 1; m++)
		{
			u(N - 1, j * THREAD_SIZE + m) = G * u(N - 2, j * THREAD_SIZE + m);
			u(i * THREAD_SIZE + m, N - 1) = G * u(i * THREAD_SIZE + m, N - 2);
		}
		u(N - 1, N - 1) = G * u(N - 1, N - 2);
	}
}

void synthesis(int T)
{
	int N = 512;
	float eta = 0.0002;
	float G = 0.75;
	float rho = 0.5;

	float* dev_U1; cudaMalloc((float**)&dev_U1, THREAD_SIZE * THREAD_SIZE * GRID_SIZE * GRID_SIZE * BLOCK_SIZE * BLOCK_SIZE * sizeof(float));
	float* dev_U2; cudaMalloc((float**)&dev_U2, THREAD_SIZE * THREAD_SIZE * GRID_SIZE * GRID_SIZE * BLOCK_SIZE * BLOCK_SIZE * sizeof(float));
	float* dev_U; cudaMalloc((float**)&dev_U, THREAD_SIZE * THREAD_SIZE * GRID_SIZE * GRID_SIZE * BLOCK_SIZE * BLOCK_SIZE * sizeof(float));
	GpuTimer timer; GpuTimer timer_total;
	timer_total.Start();
	for (int iter = 0; iter < T; iter++)
	{
		cudaMemcpy(dev_U1, u1, THREAD_SIZE * THREAD_SIZE * GRID_SIZE * GRID_SIZE * BLOCK_SIZE * BLOCK_SIZE * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_U2, u2, THREAD_SIZE * THREAD_SIZE * GRID_SIZE * GRID_SIZE * BLOCK_SIZE * BLOCK_SIZE * sizeof(float), cudaMemcpyHostToDevice);

		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); // threads are BLOCK_SIZE*BLOCK_SIZE, 32*32 1024 in my case
		dim3 dimGrid(GRID_SIZE, GRID_SIZE); // 4*4 16 blocks in a grid
		timer.Start();
		calculation << <dimGrid, dimBlock >> > (array2D<float>(dev_U1, BLOCK_SIZE*GRID_SIZE*THREAD_SIZE),
			array2D<float>(dev_U2, BLOCK_SIZE*GRID_SIZE*THREAD_SIZE),
			array2D<float>(dev_U, BLOCK_SIZE*GRID_SIZE*THREAD_SIZE),
			N,
			eta,
			rho,
			G);
		timer.Stop();
		cudaDeviceSynchronize();

		cudaMemcpy(u, dev_U,  THREAD_SIZE * THREAD_SIZE * GRID_SIZE * GRID_SIZE * BLOCK_SIZE * BLOCK_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

		// Copy the new matrix to u1
		// Update u1 u2
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < N; j++)
			{
				u2[i][j] = u1[i][j];
				u1[i][j] = u[i][j];
			}
		}

		printf("In iteration %d, u[%d][%d] is %f, running time is %fms\n", iter, N / 2, N / 2, u[N / 2][N / 2], timer.Elapsed());

	}
	timer_total.Stop();
	printf("Total time for %d threads per block, %d blocks and %d finite elements per thread is %fms\n", BLOCK_SIZE * BLOCK_SIZE, GRID_SIZE * GRID_SIZE,
		THREAD_SIZE * THREAD_SIZE, timer_total.Elapsed());

	cudaFree(dev_U1);
	cudaFree(dev_U2);
	cudaFree(dev_U);

}

int main(int argc, char** argv) {
	char* t = argv[1];
	int T = atoi(t);
	// First hit get u1 for the first step. Make u1[2][2] to 1, others to 0
	int N = 512;
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			u[i][j] = 0;
			u1[i][j] = 0;
			u2[i][j] = 0;
		}
	}
	u1[N/2][N/2] = 1.0;
	synthesis(T);
	return 0;
}