#include "gputimer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

__device__ int gate_solver(int gate, int a, int b) {
	int out = 0;
	if (gate == 0) {
		out = a & b;
	}
	else if (gate == 1) {
		out = a | b;
	}
	else if (gate == 2) {
		out = !(a & b);
	}
	else if (gate == 3) {
		out = !(a | b);
	}
	else if (gate == 4) {
		out = a ^ b;
	}
	else if (gate == 5) {
		out = !(a ^ b);
	}
	return out;
}

__global__ void global_queuing_bfs_kernel(
	int nodePtrsLength, int nodeNeighborsLength, int nodeVisitedLength, int nodeCurrentLevelLength
	, int* nodePtrs, int* nodeNeighbors
	, int* nodeVisited, int* nodeGates, int* nodeInput, int* nodeOutput
	, int* nodeCurrentLevel
	, int* nextLevel
	, int size
	, int* numNextLevelNodes
)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	for (; i < nodeCurrentLevelLength; i += size) {
		{
			int node = (nodeCurrentLevel[i]);
			for (int nbrIdx = (nodePtrs[node]); nbrIdx < (nodePtrs[node + 1]); nbrIdx++)
			{
				int neighbor = (nodeNeighbors[nbrIdx]);
				__syncthreads();
				if (atomicCAS(&nodeVisited[neighbor], 0, 1) == 0)
				{
					nodeOutput[neighbor] = gate_solver(nodeGates[neighbor], nodeOutput[node], nodeInput[neighbor]);
					atomicExch(nextLevel + (atomicAdd(numNextLevelNodes, 1)), neighbor);
				}
			}
		}
	}
}

void calculation(int BLOCK_SIZE, int NUM_BLOCKS, int argc, char* argv[])
{

	if (argc != 7) {
		printf("The input arguments should follow this format:\n./sequential <path_to_input_1.raw> <path_to_input_2.raw> <path_to_input_3.raw> <path_to_input_4.raw> <output_nodeOutput_filepath> <output_nextLevelNodes_filepath>\n");
		exit(1);
	}

	/*
	argv[1]: input1
	argv[2]: input2
	argv[3]: input3
	argv[4]: input4
	argv[5]: node output file (output)
	argv[6]: next level nodes (output)
	*/

	FILE* input1;
	FILE* input2;
	FILE* input3;
	FILE* input4;
	FILE* nodeOutputResult;
	FILE* nextLevelNodesResult;
	// open files
	input1 = fopen(argv[1], "r");
	if (!input1) {
		perror("File1 open error!\n");
		exit(1);
	}
	input2 = fopen(argv[2], "r");
	if (!input2) {
		perror("File2 open error!\n");
		exit(1);
	}
	input3 = fopen(argv[3], "r");
	if (!input3) {
		perror("File3 open error!\n");
		exit(1);
	}
	input4 = fopen(argv[4], "r");
	if (!input4) {
		perror("File4 open error!\n");
		exit(1);
	}
	nodeOutputResult = fopen(argv[5], "w");
	nextLevelNodesResult = fopen(argv[6], "w");

	// read the first line of the 4 input files and get the length of file
	int length1, length2, length3, length4;
	fscanf(input1, "%d", &length1);
	fscanf(input2, "%d", &length2);
	fscanf(input3, "%d", &length3);
	fscanf(input4, "%d", &length4);
	int* input1Content;
	int* input2Content;
	int* nodeVisited;
	int* nodeGate;
	int* nodeInput;
	int* nodeOutput;
	int* input4Content;
	int* nextLevel;
	int* numNextLevelNodes;
	// Malloc memory for GPU for each array
	int size;
	cudaMallocManaged((void**)&input1Content, length1 * sizeof(int));
	cudaMallocManaged((void**)&input2Content, length2 * sizeof(int));
	cudaMallocManaged((void**)&nodeVisited, length3 * sizeof(int));
	cudaMallocManaged((void**)&nodeGate, length3 * sizeof(int));
	cudaMallocManaged((void**)&nodeInput, length3 * sizeof(int));
	cudaMallocManaged((void**)&nodeOutput, length3 * sizeof(int));
	cudaMallocManaged((void**)&input4Content, length4 * sizeof(int));
	cudaMallocManaged((void**)&nextLevel, length2 * sizeof(int));
	cudaMallocManaged((void**)&numNextLevelNodes, 1 * sizeof(int));

	// copy the input file 1 content to an array
	char* line = (char*)malloc(10);
	int i = 0;
	while (fgets(line, 10, input1))
	{
		if (i == 0)
		{
			i++;
			continue;
		}
		else
		{
			input1Content[i - 1] = atoi(line);
			i++;
		}
	}

	// copy the input file 2 content to an array
	i = 0;
	while (fgets(line, 10, input2))
	{
		if (i == 0)
		{
			i++;
			continue;
		}
		else
		{
			input2Content[i - 1] = atoi(line);
			i++;
		}
	}

	// copy the input file 3 content to an array
	i = 0;
	while (fgets(line, 10, input3))
	{
		if (i == 0)
		{
			i++;
			continue;
		}
		else
		{
			nodeVisited[i - 1] = line[0] - 48;
			nodeGate[i - 1] = line[2] - 48;
			nodeInput[i - 1] = line[4] - 48;
			if (line[6] == '-')
			{
				nodeOutput[i - 1] = -1;
			}
			else
			{
				nodeOutput[i - 1] = line[6] - 48;
			}
			i++;
		}
	}

	// copy the input file 4 content to an array
	i = 0;
	while (fgets(line, 10, input4))
	{
		if (i == 0)
		{
			i++;
			continue;
		}
		else
		{
			input4Content[i - 1] = atoi(line);
			i++;
		}
	}
	*numNextLevelNodes = 0;
	size = (NUM_BLOCKS * BLOCK_SIZE);
	printf("Parallel using blockSize %d and numBlock %d\n", BLOCK_SIZE, NUM_BLOCKS);
	GpuTimer timer;
	timer.Start();
	global_queuing_bfs_kernel << <NUM_BLOCKS, BLOCK_SIZE >> > (length1, length2, length3, length4
		, input1Content
		, input2Content
		, nodeVisited, nodeGate, nodeInput
		, nodeOutput
		, input4Content
		, nextLevel
		, size
		, numNextLevelNodes);
	timer.Stop();
	cudaDeviceSynchronize();
	printf("Time for global queueing: %f ms\n", timer.Elapsed());
	i = 0;
	fprintf(nextLevelNodesResult, "%d\n", *numNextLevelNodes);
	while (i < *numNextLevelNodes)
	{
		fprintf(nextLevelNodesResult, "%d\n", nextLevel[i]);
		i++;
	}
	i = 0;
	fprintf(nodeOutputResult, "%d\n", length3);
	while (i < length3)
	{
		fprintf(nodeOutputResult, "%d\n", nodeOutput[i]);
		i++;
	}
	cudaFree(input1Content);
	cudaFree(input2Content);
	cudaFree(nodeVisited);
	cudaFree(nodeGate);
	cudaFree(nodeInput);
	cudaFree(nodeOutput);
	cudaFree(input4Content);
	cudaFree(nextLevel);
	cudaFree(numNextLevelNodes);
	free(line);
	fclose(input1);
	fclose(input2);
	fclose(input3);
	fclose(input4);
	fclose(nodeOutputResult);
	fclose(nextLevelNodesResult);
}

int main(int argc, char* argv[]) {
	int BLOCK_SIZE[3] = { 32, 64, 128 };
	int NUM_BLOCKS[3] = { 10, 25, 35 };
	for (int i = 0; i < 3; i++) {
		calculation(BLOCK_SIZE[i], NUM_BLOCKS[i], argc, argv);
	}

	return 0;
}