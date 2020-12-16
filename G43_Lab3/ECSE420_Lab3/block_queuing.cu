
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gputimer.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>


__device__ int gate_solve(int gate, int a, int b) {
    int i = 0; 
    if (gate == 0) { 
        i = a && b;
    }
    else if (gate == 1) { 
        i = a || b;
    }
    else if (gate == 2) { 
        i = !(a && b);
    }
    else if (gate == 3) {  
        i = !(a || b);
    }
    else if (gate == 4) {  
        i = (a || b) && (!a || !b); 
    }
    else if (gate == 5) { 
        i = !((a || b) && (!a || !b)); 
    }
    return i; 
}


__global__ void global_queuing_bfs_kernel(
    int nodePtrsLength, int nodeNeighborsLength, int nodeVisitedLength, int nodeCurrentLevelLength
	, int* nodePtrs, int* nodeNeighbors
	, int* nodeVisited, int* nodeGates, int* nodeInput, int* nodeOutput
	, int* nodeCurrentLevel
  , int* nextLevel
  , int size
  , int* numNextLevelNodes
  , int queueCap
  , int *num) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ int blockQueueNLs[];
    __shared__ int counter;
  for(; i < nodeCurrentLevelLength; i += size) {
      int node = nodeCurrentLevel[i];
      for(int nbridx = nodePtrs[node]; nbridx < nodePtrs[node+1]; nbridx++) {
          int neighbor = nodeNeighbors[nbridx]; 
          if (atomicCAS(&nodeVisited[neighbor], 0, 1) == 0) {
              nodeOutput[neighbor] = gate_solve(nodeGates[neighbor], nodeOutput[node], nodeInput[neighbor]);
              int index = atomicAdd(&counter, 1);
              if (index < queueCap) {
                  atomicExch(&(blockQueueNLs[index]), neighbor);
              }
              else {
                  atomicExch(&counter, queueCap);
                  atomicExch(nextLevel+atomicAdd(numNextLevelNodes, 1), neighbor);
              }       
          }
      }
  }
    __syncthreads();

    if(threadIdx.x == 0) {
      for(int j = 0; j < counter; j++) {
        atomicExch(nextLevel+atomicAdd(numNextLevelNodes, 1), blockQueueNLs[j]);
      }
    }
}

int process(int argc, char* argv[], int blockSize, int blockNum, int queueCap) {
    struct GpuTimer timer;

    /*
    argv[4]: input1
    argv[5]: input2
    argv[6]: input3
    argv[7]: input4
    argv[8]: node output file (output)
    argv[9]: next level nodes (output)
    */
    FILE* input1;
    FILE* input2;
    FILE* input3;
    FILE* input4;
    FILE* nodeOutputResult;
    FILE* nextLevelNodesResult;
	  printf("start\n");
    // open files
    input1 = fopen(argv[4], "r");
    if (!input1) {
        perror("File open error!\n");
        return 1;
    }
    input2 = fopen(argv[5], "r");
    if (!input2) {
        perror("File open error!\n");
        return 1;
    }
    input3 = fopen(argv[6], "r");
    if (!input3) {
        perror("File open error!\n");
        return 1;
    }
    input4 = fopen(argv[7], "r");
    if (!input4) {
        perror("File open error!\n");
        return 1;
    }
    nodeOutputResult = fopen(argv[8], "w");
    nextLevelNodesResult = fopen(argv[9], "w");
	
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
    int* num;
    // Malloc memory for GPU for each array
    int size = (blockNum*blockSize);
    cudaMallocManaged((void**)&input1Content, length1 * sizeof(int));
    cudaMallocManaged((void**)&input2Content, length2 * sizeof(int));
	  cudaMallocManaged((void**)&nodeVisited, length3 * sizeof(int));
    cudaMallocManaged((void**)&nodeGate, length3 * sizeof(int));
    cudaMallocManaged((void**)&nodeInput, length3 * sizeof(int));
    cudaMallocManaged((void**)&nodeOutput, length3 * sizeof(int));
	  cudaMallocManaged((void**)&input4Content, length4 * sizeof(int));
    cudaMallocManaged((void**)&nextLevel, length3 * sizeof(int)); // length2 and length3 both can do the job
    cudaMallocManaged((void**)&numNextLevelNodes, 1*sizeof(int));
    cudaMallocManaged((void**)&num, 1*sizeof(int));
    
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
    int temp1;
    int temp2;
    int temp3;
    int temp4;
    while (fscanf(input3, "%d,%d,%d,%d", &temp1, &temp2, &temp3, &temp4) == 4) {
        nodeVisited[i] = temp1;
        nodeGate[i] = temp2;
        nodeInput[i] = temp3;
        nodeOutput[i] = temp4;
        i++;
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
    num = 0;
    timer.Start();
    global_queuing_bfs_kernel <<<blockNum, blockSize, queueCap*4>>> (length1, length2, length3, length4
                                                            , input1Content
                                                            , input2Content
                                                            , nodeVisited, nodeGate, nodeInput, nodeOutput
                                                            , input4Content
                                                            , nextLevel
                                                            , size
                                                            , numNextLevelNodes
                                                            , queueCap
                                                            , num);
    timer.Stop();
    cudaDeviceSynchronize();

    printf("Time for block queueing: %f ms with block numbers: %d, threads: %d, queue capacity: %d\n"
          , timer.Elapsed()/1000, blockNum, blockSize, queueCap);
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
    cudaFree(num);
    free(line);
    fclose(input1);
    fclose(input2);
    fclose(input3);
    fclose(input4);
    fclose(nodeOutputResult);
    fclose(nextLevelNodesResult);
    return 0;
}

int main(int argc, char* argv[])
{

        process(argc, argv, atoi(argv[1]), atoi(argv[2]), atoi(argv[3]));
      
}