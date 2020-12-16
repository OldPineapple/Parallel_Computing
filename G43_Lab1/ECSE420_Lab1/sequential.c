#include <stdio.h>
#include <stdlib.h>
// #include <cuda_runtime.h>
// #include <device_launch_parameters.h>
#include <time.h>
// #include <ctime>
#define AND 0
#define OR 1
#define NAND 2
#define NOR 3
#define XOR 4
#define XNOR 5

void processData(char* data, int length, char* results)
{
	for(int i=0; i<length/3; i++)
	{
		// a = data[i];
		// b = data[i + 2];
		// opCode = data[i + 4];
		// in ascii code 1 is 49 and 0 is 48
		if(data[i*3 + 4] == '0')
		{
			int result = (((data[i*3]) - '0') & ((data[i*3 + 2]) - '0'));
			results[i] = (result + '0');
			i++;
			results[i] = '\n';
		}
		if(data[i*3 + 4] == '1')
		{
			int result = ((data[i*3]) - '0') | ((data[i*3 + 2]) - '0');
			results[i] = (result + '0');
			i++;
			results[i] = '\n';
		}
		if(data[i*3 + 4] == '2')
		{
			int result = !(((data[i*3]) - '0') & ((data[i*3 + 2]) - '0'));
			results[i] = (result + '0');
			i++;
			results[i] = '\n';
		}
		if(data[i*3 + 4] == '3')
		{
			int result = !(((data[i*3]) - '0') | ((data[i*3 + 2]) - '0'));
			results[i] = (result + '0');
			i++;
			results[i] = '\n';
		}
		if(data[i*3 + 4] == '4')
		{
			int result = (((data[i*3]) - '0') ^ ((data[i*3 + 2]) - '0'));
			results[i] = (result + '0');
			i++;
			results[i] = '\n';
		}
		if(data[i*3 + 4] == '5')
		{
			int result = !(((data[i*3]) - '0') ^ ((data[i*3 + 2]) - '0'));
			results[i] = (result + '0');
			i++;
			results[i] = '\n';
		}
	}
}

void parseFile(FILE* fp, int length, FILE* output)
{

	char* data;
	char* results;
	data = (char*)malloc(length);
	results = (char*)malloc(length / 3);
	fread(data, 1, length, fp);
	clock_t start = clock();
	processData(data, length, results);
	clock_t end = clock();
	fputs(results, output);
	// printf("Clocks per second == %f ", CLOCKS_PER_SEC);
	printf("Time used: %f ms\n", (float)(end-start)/CLOCKS_PER_SEC * 1000);
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
		exit(EXIT_FAILURE);

	char* outputFileName = argv[3];
	FILE* output = fopen(outputFileName, "w");
	if (output == NULL)
		exit(EXIT_FAILURE);

	parseFile(input, atoi(argv[2])*6, output);

	fclose(input);
	fclose(output);

	return 0;

}