#include "gputimer.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <ctime>

#define AND 0
#define OR 1
#define NAND 2
#define NOR 3
#define XOR 4
#define XNOR 5

__global__ void classify(char* d_data, int SIZE, char* d_results) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	// since only needs to launch one thread per logic gate, SIZE should equal to the number of rows in the file
	if (i < SIZE) {
		if (d_data[i * 6 + 4] == '0')
		{
			int result = (((d_data[i * 6]) - '0') & ((d_data[i * 6 + 2]) - '0'));
			d_results[2 * i] = (result + '0');
			d_results[2 * i + 1] = '\n';
		}
		else if (d_data[i * 6 + 4] == '1')
		{
			int result = ((d_data[i * 6]) - '0') | ((d_data[i * 6 + 2]) - '0');
			d_results[2 * i] = (result + '0');
			d_results[2 * i + 1] = '\n';
		}
		else if (d_data[i * 6 + 4] == '2')
		{
			int result = !(((d_data[i * 6]) - '0') & ((d_data[i * 6 + 2]) - '0'));
			d_results[2 * i] = (result + '0');
			d_results[2 * i + 1] = '\n';
		}
		else if (d_data[i * 6 + 4] == '3')
		{
			int result = !(((d_data[i * 6]) - '0') | ((d_data[i * 6 + 2]) - '0'));
			d_results[2 * i] = (result + '0');
			d_results[2 * i + 1] = '\n';
		}
		else if (d_data[i * 6 + 4] == '4')
		{
			int result = (((d_data[i * 6]) - '0') ^ ((d_data[i * 6 + 2]) - '0'));
			d_results[2 * i] = (result + '0');
			d_results[2 * i + 1] = '\n';
		}
		else if (d_data[i * 6 + 4] == '5')
		{
			int result = !(((d_data[i * 6]) - '0') ^ ((d_data[i * 6 + 2]) - '0'));
			d_results[2 * i] = (result + '0');
			d_results[2 * i + 1] = '\n';
		}
	}
}

void parallel_explicit(FILE* fp_in, int length, FILE* fp_out) {
	// input has length 'length' and output has length 'length/3'
	// output file has only 2 elements in one line (a number and a '\n') while input file has 6 (listed in main)
	char* data, * d_data, * results, * d_results;
	// timer_kernel records time for kernel function
	// timer_migration records explicit data migration time (copy data from host to device)
	GpuTimer timer_kernel, timer_migration;
	data = (char*)malloc(length);
	results = (char*)malloc(length/3);
	cudaMalloc(&d_data, length);
	cudaMalloc(&d_results, length/3);
	fread(data, 1, length, fp_in);
	timer_migration.Start();
	cudaMemcpy(d_data, data, length, cudaMemcpyHostToDevice);
	cudaMemcpy(d_results, results, length/3, cudaMemcpyHostToDevice);
	timer_migration.Stop();
	int maxThreadNum = 1024;
	// distribute the total threads equally in blocks
	while(1)
	{
		if(length/6 % maxThreadNum != 0)
		{
			maxThreadNum--;
		}
		else
		{
			break;
		}
	}
	int totalBlocks = length / 6 / maxThreadNum;
	timer_kernel.Start();
	classify <<<totalBlocks, maxThreadNum>>> (d_data, length/6, d_results);
	timer_kernel.Stop();
	cudaMemcpy(data, d_data, length, cudaMemcpyDeviceToHost);
	cudaMemcpy(results, d_results, length/3, cudaMemcpyDeviceToHost);
	printf("Time for kernel functions: %f ms\n", timer_kernel.Elapsed());
	printf("Time for explicit data migration: %f ms\n", timer_migration.Elapsed());

	fputs(results, fp_out);
	
	cudaFree(d_data);
	cudaFree(d_results);
	free(data);
	free(results);
}

int main(int argc, char* argv[])
{
	if ( argc < 4)
	{
		printf("You must enter 3 input files!\n");
		exit(1);
	}

	// argv[1] : input_file_path
	// argv[2] : input_file_length
	// argv[3] : output_file_path

	char* fileName = argv[1];
	FILE* input = fopen(fileName, "r");
	if (input == NULL)
	{
		exit(EXIT_FAILURE);
		printf("No input file.\n");
	}


	char* outputFileName = argv[3];
	FILE* output = fopen(outputFileName, "w");
	if (output == NULL)
	{
		exit(EXIT_FAILURE);
		printf("No output file.\n");
	}

	// atoi(argv[2]) is the number of lines in a file
	// each line has 6 elements, including 3 numbers, 2 commas and 1 '\n'
	// as a result the length pass in should be atoi(argv[2]) * 6
	printf("For input file %s \n", fileName);
	parallel_explicit(input, atoi(argv[2]) * 6, output);
	fclose(input);
	fclose(output);

	return 0;
}