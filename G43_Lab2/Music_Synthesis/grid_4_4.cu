
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define BLOCK_SIZE 4
#define GRID_SIZE 1
float u1[BLOCK_SIZE][BLOCK_SIZE];
float u2[BLOCK_SIZE][BLOCK_SIZE];
float u[BLOCK_SIZE][BLOCK_SIZE];

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

	if (0 < i && i < N - 1 && 0 < j && j < N - 1)
	{
		u(i, j) = (rho * (u1(i - 1, j) + u1(i + 1, j) + u1(i, j - 1) + u1(i, j + 1) - 4 * u1(i,j)) + 2 * u1(i,j) - (1 - eta) * u2(i,j)) / (1 + eta);
	}

	__syncthreads();

	if (i == N - 1 && j != 0 && j != N - 1)
	{
		u(N - 1, j) = G * u(N - 2, j);
	}
	else if (i == 0 && j != 0 && j != N - 1)
	{
		u(0, j) = G * u(1, j);
	}
	else if (i != 0 && i != N - 1 && j == 0)
	{
		u(i, 0) = G * u(i, 1);
	}
	else if (i != 0 && i != N - 1 && j == N - 1)
	{
		u(i, N - 1) = G * u(i, N - 2);
	}
	__syncthreads();

	if (i == 0 && j == 0)
	{
		u(0, 0) = G * u(1, 0);
	}
	else if (j == 0 && i == N - 1)
	{
		u(N - 1, 0) = G * u(N - 2, 0);
	}
	else if (i == 0 && j == N - 1)
	{
		u(0, N - 1) = G * u(0, N - 2);
	}
	else if (i == N - 1 && j == N - 1)
	{
		u(N - 1, N - 1) = G * u(N - 1, N - 2);
	}
}

void synthesis(int T)
{
	int N = 4;
	float eta = 0.0002;
	float G = 0.75;
	float rho = 0.5;

	float* dev_U1; cudaMalloc((float**)&dev_U1, BLOCK_SIZE * BLOCK_SIZE * sizeof(float));
	float* dev_U2; cudaMalloc((float**)&dev_U2, BLOCK_SIZE * BLOCK_SIZE * sizeof(float));
	float* dev_U; cudaMalloc((float**)&dev_U, BLOCK_SIZE * BLOCK_SIZE * sizeof(float));

	for (int iter = 0; iter < T; iter++)
	{
		cudaMemcpy(dev_U1, u1, BLOCK_SIZE * BLOCK_SIZE * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_U2, u2, BLOCK_SIZE * BLOCK_SIZE * sizeof(float), cudaMemcpyHostToDevice);

		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); // threads are BLOCK_SIZE*BLOCK_SIZE, 4*4 16 in my case
		dim3 dimGrid(GRID_SIZE, GRID_SIZE); // 1*1 blocks in a grid

		calculation << <dimGrid, dimBlock >> > (array2D<float>(dev_U1, BLOCK_SIZE),
			array2D<float>(dev_U2, BLOCK_SIZE),
			array2D<float>(dev_U, BLOCK_SIZE),
			N,
			eta,
			rho,
			G);

		cudaDeviceSynchronize();

		cudaMemcpy(u, dev_U, BLOCK_SIZE * BLOCK_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

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

		printf("In iteration %d, u[%d][%d] is %f\n", iter, N / 2, N / 2, u[N / 2][N / 2]);

	}

	cudaFree(dev_U1);
	cudaFree(dev_U2);
	cudaFree(dev_U);

}

int main(int argc, char** argv) {
	char* t = argv[1];
	int T = atoi(t);
	// First hit get u1 for the first step. Make u1[2][2] to 1, others to 0
	int N = 4;
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			u[i][j] = 0;
			u1[i][j] = 0;
			u2[i][j] = 0;
		}
	}
	u1[2][2] = 1.0;
	synthesis(T);
	return 0;
}