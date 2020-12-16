/* Example of using lodepng to load, process, save image */
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "gputimer.h"
#include "lodepng.h"
#include "wm - Copy.h"

__global__ void convolve(unsigned char* input_image, unsigned char* output_image, unsigned width, int times, int thread_size) {
	float lt, mt, rt, l, m, r, lb, mb, rb;
	float sum;
	unsigned char output_pixel;
	int index = threadIdx.x;
	unsigned int pixel_number_in_array = 0; // This is the current pixel number in the array formed by image pixels.
	float w[3][3] =
	{
	  1,	2,		-1,
	  2,	0.25,	-2,
	  1,	-2,		-1
	};
	if (index < thread_size) {
		pixel_number_in_array = times * thread_size + index; // (times*thread_size + index): output current thread number
		pixel_number_in_array = (pixel_number_in_array / (width - 2)) * width + pixel_number_in_array % (width - 2);
		// R
		lt = (float)input_image[pixel_number_in_array * 4] * w[0][0];
		mt = (float)input_image[(pixel_number_in_array + 1) * 4] * w[0][1];
		rt = (float)input_image[(pixel_number_in_array + 2) * 4] * w[0][2];
		l = (float)input_image[(pixel_number_in_array + width) * 4] * w[1][0];
		m = (float)input_image[(pixel_number_in_array + width + 1) * 4] * w[1][1];
		r = (float)input_image[(pixel_number_in_array + width + 2) * 4] * w[1][2];
		lb = (float)input_image[(pixel_number_in_array + 2 * width) * 4] * w[2][0];
		mb = (float)input_image[(pixel_number_in_array + 2 * width + 1) * 4] * w[2][1];
		rb = (float)input_image[(pixel_number_in_array + 2 * width + 2) * 4] * w[2][2];
		sum = lt + mt + rt + l + m + r + lb + mb + rb;
		if (sum < 0)
			sum = 0;
		else if (sum > 255)
			sum = 255;
		if ((sum - (int)sum) >= 0.5)
			sum = sum + 1;
		output_pixel = (unsigned char)sum;
		output_image[(thread_size * times + index) * 4 + 0] = output_pixel;

		// G
		lt = (float)input_image[pixel_number_in_array * 4 + 1] * w[0][0];
		mt = (float)input_image[(pixel_number_in_array + 1) * 4 + 1] * w[0][1];
		rt = (float)input_image[(pixel_number_in_array + 2) * 4 + 1] * w[0][2];
		l = (float)input_image[(pixel_number_in_array + width) * 4 + 1] * w[1][0];
		m = (float)input_image[(pixel_number_in_array + width + 1) * 4 + 1] * w[1][1];
		r = (float)input_image[(pixel_number_in_array + width + 2) * 4 + 1] * w[1][2];
		lb = (float)input_image[(pixel_number_in_array + 2 * width) * 4 + 1] * w[2][0];
		mb = (float)input_image[(pixel_number_in_array + 2 * width + 1) * 4 + 1] * w[2][1];
		rb = (float)input_image[(pixel_number_in_array + 2 * width + 2) * 4 + 1] * w[2][2];
		sum = lt + mt + rt + l + m + r + lb + mb + rb;
		if (sum < 0)
			sum = 0;
		else if (sum > 255)
			sum = 255;
		if ((sum - (int)sum) >= 0.5)
			sum = sum + 1;
		output_pixel = (unsigned char)sum;
		output_image[(thread_size * times + index) * 4 + 1] = output_pixel;

		// B
		lt = (float)input_image[pixel_number_in_array * 4 + 2] * w[0][0];
		mt = (float)input_image[(pixel_number_in_array + 1) * 4 + 2] * w[0][1];
		rt = (float)input_image[(pixel_number_in_array + 2) * 4 + 2] * w[0][2];
		l = (float)input_image[(pixel_number_in_array + width) * 4 + 2] * w[1][0];
		m = (float)input_image[(pixel_number_in_array + width + 1) * 4 + 2] * w[1][1];
		r = (float)input_image[(pixel_number_in_array + width + 2) * 4 + 2] * w[1][2];
		lb = (float)input_image[(pixel_number_in_array + 2 * width) * 4 + 2] * w[2][0];
		mb = (float)input_image[(pixel_number_in_array + 2 * width + 1) * 4 + 2] * w[2][1];
		rb = (float)input_image[(pixel_number_in_array + 2 * width + 2) * 4 + 2] * w[2][2];
		sum = lt + mt + rt + l + m + r + lb + mb + rb;
		if (sum < 0)
			sum = 0;
		else if (sum > 255)
			sum = 255;
		if ((sum - (int)sum) >= 0.5)
			sum = sum + 1;
		output_pixel = (unsigned char)sum;
		output_image[(thread_size * times + index) * 4 + 2] = output_pixel;

		// A
		output_pixel = (unsigned char)input_image[(pixel_number_in_array + width + 1) * 4 + 3];
		output_image[(thread_size * times + index) * 4 + 3] = output_pixel;
	}
}

void image_convolution(char* input_filename, char* output_filename)
{
	unsigned error;
	unsigned char* image, * output_image, * input_image;
	unsigned width, height;

	error = lodepng_decode32_file(&image, &width, &height, input_filename);
	if (error) printf("error %u: %s\n", error, lodepng_error_text(error));
	size_t input_image_size = width * height * 4 * sizeof(unsigned char);
	size_t output_image_size = (width - 2) * (height - 2) * 4 * sizeof(unsigned char);
	// process image
	int size[10] = { 1, 4, 8, 16, 32, 64, 128, 256, 512, 1024 };
	for (int s = 0; s < 10; s++) {
		printf("Convolve image %s using %d thread(s): ", input_filename, size[s]);
		// Set timer
		GpuTimer timer;
		// allocate GPU memory
		cudaMallocManaged((void**)&input_image, input_image_size);
		cudaMallocManaged((void**)&output_image, output_image_size);
		// initialize GPU memory
		for (int i = 0; i < input_image_size; i++) {
			input_image[i] = image[i];
		}
		for (int i = 0; i < output_image_size; i++) {
			output_image[i] = 0;
		}
		// each time run size[s] threads until task finished
		int total_pixel = (width - 2) * (height - 2); // for output
		int quotient = (total_pixel + size[s] - 1) / size[s];
		// start timer
		timer.Start();
		for (int j = 0; j < quotient; j++) {
			convolve <<<1, size[s] >>> (input_image, output_image, width, j, size[s]);
		}
		timer.Stop();
		cudaDeviceSynchronize();
		printf("%fseconds\n", timer.Elapsed() / 1000); // Convert unit from ms to s
		lodepng_encode32_file(output_filename, output_image, (width - 2) , (height - 2));
		cudaFree(input_image);
		cudaFree(output_image);
	}
}

int main(void)
{
	char* input_filename1 = "Test_1.png";
	char* input_filename2 = "Test_2.png";
	char* input_filename3 = "Test_3.png";
	char* output_filename1 = "Test_1_convolution.png";
	char* output_filename2 = "Test_2_convolution.png";
	char* output_filename3 = "Test_3_convolution.png";
	image_convolution(input_filename1, output_filename1);
	image_convolution(input_filename2, output_filename2);
	image_convolution(input_filename3, output_filename3);
	return 0;
}
