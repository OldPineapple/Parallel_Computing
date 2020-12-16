/* Example of using lodepng to load, process, save image */
#include "lodepng.h"
#include "gputimer.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void rectify(unsigned char *input_image, unsigned char *output_image, int times, int thread_size) {
	// Use parallel threads
	int index = threadIdx.x;

	if (index < thread_size) {
		if (input_image[(thread_size * times + index) * 4 + 0] >= 127) // R
			output_image[(thread_size * times + index) * 4 + 0] = input_image[(thread_size * times + index) * 4 + 0];
		else
			output_image[(thread_size * times + index) * 4 + 0] = 127;
		if (input_image[(thread_size * times + index) * 4 + 1] >= 127) // G
			output_image[(thread_size * times + index) * 4 + 1] = input_image[(thread_size * times + index) * 4 + 1];
		else
			output_image[(thread_size * times + index) * 4 + 1] = 127;
		if (input_image[(thread_size * times + index) * 4 + 2] >= 127) // B
			output_image[(thread_size * times + index) * 4 + 2] = input_image[(thread_size * times + index) * 4 + 2];
		else
			output_image[(thread_size * times + index) * 4 + 2] = 127;
		output_image[(thread_size * times + index) * 4 + 3] = input_image[(thread_size * times + index) * 4 + 3]; // A
	}
}

__global__ void pool(unsigned char *input_image, unsigned char *output_image, unsigned width, int times , int thread_size) {
	unsigned char lb, rb, lt, rt, max;
	int index = threadIdx.x;
	unsigned pixel_number_in_array = 0; // This is the current pixel number in the array formed by image pixels.

	if (index < thread_size) {
		for (int i = 0; i < 4; i++) {

			pixel_number_in_array = (times * thread_size + index) * 2; // (times*thread_size + index): output current thread number; *2 since # of input width / # of output width = 2
			pixel_number_in_array += (width * (pixel_number_in_array / width)); // pixel_number_in_array/width represents how many rows we have passed(round down). 
			// If we just multiply by 2 then we would only increase in one dimension (X) and miss the increase in the other dimension (Y). 
			// The increase in the other dimension is like a width scale that is we want to increase 1 in Y dimension we need to increase width units in X dimension
			// So we need to use PixelNumberInArray/width cast to int to know how many unit we should increase to Y.
			lb = input_image[pixel_number_in_array * 4 + i];
			rb = input_image[(pixel_number_in_array + 1) * 4 + i];
			lt = input_image[(pixel_number_in_array + width) * 4 + i];
			rt = input_image[(pixel_number_in_array + width + 1) * 4 + i];
			max = lb;
			if (rb > max)
				max = rb;
			if (lt > max)
				max = lt;
			if (rt > max)
				max = rt;

			output_image[(thread_size * times + index) * 4 + i] = max;
		}
	}
}

void image_rectification(char* input_filename, char* output_filename)
{
	unsigned error;
	unsigned char *image, *output_image, *input_image;
	unsigned width, height;

	error = lodepng_decode32_file(&image, &width, &height, input_filename);
	if (error) printf("error %u: %s\n", error, lodepng_error_text(error));
	size_t image_size = width * height * 4 * sizeof(unsigned char);
	// process image
	int size[9] = { 1, 2, 4, 8, 16, 32, 64, 128, 256 };
	for (int s = 0; s < 9; s++) {
		printf("Rectify image %s using %d thread(s): ", input_filename, size[s]);
		// Set timer
		GpuTimer timer;
		// allocate GPU memory
		cudaMallocManaged((void**)&input_image, image_size);
		cudaMallocManaged((void**)&output_image, image_size);
		// initialize GPU memory
		for (int i = 0; i < image_size; i++) {
			output_image[i] = 0;
			input_image[i] = image[i];
		}
		// each time run size[s] threads until task finished
		int total = width * height;
		int quotient = (total + size[s] - 1) / size[s];
		// start timer
		timer.Start();
		for (int j = 0; j < quotient; j++) {
			rectify <<<1, size[s] >>> (input_image, output_image, j, size[s]);
		}
		cudaDeviceSynchronize();
		timer.Stop();
		printf("%fseconds\n", timer.Elapsed() / 1000); // Convert unit from ms to s
		lodepng_encode32_file(output_filename, output_image, width, height);
		cudaFree(input_image);
		cudaFree(output_image);
	}
	free(image);
}

void image_pooling(char* input_filename, char* output_filename)
{
	unsigned error;
	unsigned char *image, *output_image, *input_image;
	unsigned width, height;

	error = lodepng_decode32_file(&image, &width, &height, input_filename);
	if (error) printf("error %u: %s\n", error, lodepng_error_text(error));
	size_t image_size = width * height * 4 * sizeof(unsigned char);
	// process image
	int size[9] = { 1, 2, 4, 8, 16, 32, 64, 128, 256 };
	for (int s = 0; s < 9; s++) {
		printf("Pool image %s using %d thread(s): ", input_filename, size[s]);
		// Set timer
		GpuTimer timer;
		// allocate GPU memory
		cudaMallocManaged((void**)&input_image, image_size);
		cudaMallocManaged((void**)&output_image, image_size/4);
		// initialize GPU memory
		for (int i = 0; i < image_size; i++) {
			input_image[i] = image[i];
		}
		for (int i = 0; i < image_size/4; i++) {
			output_image[i] = 0;
		}
		// each time run size[s] threads until task finished
		int total_pixel = width * height / 4; // for output
		int quotient = (total_pixel + size[s] - 1) / size[s];
		// start timer
		timer.Start();
		for (int j = 0; j < quotient; j++) {
			pool <<<1, size[s] >>> (input_image, output_image, width, j, size[s]);
		}
		cudaDeviceSynchronize();
		timer.Stop();
		printf("%fseconds\n", timer.Elapsed() / 1000); // Convert unit from ms to s
		lodepng_encode32_file(output_filename, output_image, width/2, height/2);
		cudaFree(input_image);
		cudaFree(output_image);
	}
	free(image);
}
 
int main(void)
{
  char* input_filename1 = "Test_1.png";
  char* input_filename2 = "Test_2.png";
  char* input_filename3 = "Test_3.png";
  char* output_filename1 = "Test_1_rectify.png";
  char* output_filename2 = "Test_2_rectify.png";
  char* output_filename3 = "Test_3_rectify.png";
  char* output_filename4 = "Test_1_pool.png";
  char* output_filename5 = "Test_2_pool.png";
  char* output_filename6 = "Test_3_pool.png";
  image_rectification(input_filename1, output_filename1);
  image_rectification(input_filename2, output_filename2);
  image_rectification(input_filename3, output_filename3);
  image_pooling(input_filename1, output_filename4);
  image_pooling(input_filename2, output_filename5);
  image_pooling(input_filename3, output_filename6);
  return 0;
}
