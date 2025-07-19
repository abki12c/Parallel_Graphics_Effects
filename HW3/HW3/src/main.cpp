#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <CL/cl.h>
#include <array>
#include <omp.h>

const int KERNEL_RADIUS = 8;
const float sigma = 3.f;


unsigned char blurAxis(int x, int y, int channel, int axis/*0: horizontal axis, 1: vertical axis*/, unsigned char* input, int width, int height)
{
	float sum_weight = 0.0f;
	float ret = 0.f;

	for (int offset = -KERNEL_RADIUS; offset <= KERNEL_RADIUS; offset++)
	{
		int offset_x = axis == 0 ? offset : 0;
		int offset_y = axis == 1 ? offset : 0;
		int pixel_y = std::max(std::min(y + offset_y, height - 1), 0);
		int pixel_x = std::max(std::min(x + offset_x, width - 1), 0);
		int pixel = pixel_y * width + pixel_x;

		float weight = std::exp(-(offset * offset) / (2.f * sigma * sigma));

		ret += weight * input[4 * pixel + channel];
		sum_weight += weight;
	}
	ret /= sum_weight;
	return (unsigned char)std::max(std::min(ret, 255.f), 0.f);
}

const char* loadKernelFromFile(const char* filename)
{
	FILE* file = fopen(filename, "r");
	if (!file) {
		std::cerr << "Failed to open kernel file: " << filename << std::endl;
		exit(1);
	}
	fseek(file, 0, SEEK_END);
	size_t size = ftell(file);
	rewind(file);
	char* source = new char[size + 1];
	fread(source, 1, size, file);
	source[size] = '\0';
	fclose(file);
	return source;
}

void check_error(cl_int err)
{
	if (err != CL_SUCCESS) {
		std::cerr << "OpenCL Error at " << __FILE__ << "Error code: " << err << std::endl;
		std::exit(EXIT_FAILURE);
	}
}

void gaussian_blur_separate_serial(const char* filename)
{
	int width = 0;
	int height = 0;
	int img_orig_channels = 4;
	// Load an image into an array of unsigned chars that is the size of width * height * number of channels. The channels are the Red, Green, Blue and Alpha channels of the image.
	unsigned char* img_in = stbi_load(filename, &width, &height, &img_orig_channels /*image file channels*/, 4 /*requested channels*/);
	if (img_in == nullptr)
	{
		printf("Could not load %s\n", filename);
		return;
	}

	unsigned char* img_horizontal_blur = new unsigned char[width * height * 4];
	unsigned char* img_out = new unsigned char[width * height * 4];

	// Timer to measure performance
	auto start = std::chrono::high_resolution_clock::now();

	// Horizontal Blur
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			int pixel = y * width + x;
			for (int channel = 0; channel < 4; channel++)
			{
				img_horizontal_blur[4 * pixel + channel] = blurAxis(x, y, channel, 0, img_in, width, height);
			}
		}
	}
	// Vertical Blur
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			int pixel = y * width + x;
			for (int channel = 0; channel < 4; channel++)
			{
				img_out[4 * pixel + channel] = blurAxis(x, y, channel, 1, img_horizontal_blur, width, height);
			}
		}
	}
	// Timer to measure performance
	auto end = std::chrono::high_resolution_clock::now();
	// Computation time in milliseconds
	int time = (int)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	printf("Gaussian Blur Separate - Serial: Time %dms\n", time);

	// Write the blurred image into a JPG file
	stbi_write_jpg("images/blurred_separate.jpg", width, height, 4/*channels*/, img_out, 90 /*quality*/);

	stbi_image_free(img_in);
	delete[] img_horizontal_blur;
	delete[] img_out;
}


void gaussian_blur_separate_parallel(const char* filename) 
{
	int width = 0;
	int height = 0;
	int img_orig_channels = 4;
	// Load an image into an array of unsigned chars that is the size of width * height * number of channels. The channels are the Red, Green, Blue and Alpha channels of the image.
	unsigned char* img_in = stbi_load(filename, &width, &height, &img_orig_channels /*image file channels*/, 4 /*requested channels*/);
	if (img_in == nullptr)
	{
		printf("Could not load %s\n", filename);
		return;
	}

	size_t img_size = width * height * 4;
	unsigned char* img_out = new unsigned char[img_size];


	// calculate weights
	std::array<float, 2 * KERNEL_RADIUS + 1> weights{};

	#pragma omp parallel for
	for (int offset = -KERNEL_RADIUS; offset <= KERNEL_RADIUS; offset++)
	{
		weights[offset + KERNEL_RADIUS] = std::exp(-(offset * offset) / (2.f * sigma * sigma));
	}

	// Timer to measure performance
	auto start = std::chrono::high_resolution_clock::now();

	cl_platform_id platform;

	//Set up the Platform
	cl_int error = clGetPlatformIDs(1, &platform, nullptr);
	check_error(error);

	// Set up device 
	cl_device_id device;
	error = clGetDeviceIDs(platform,CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
	check_error(error);

	// Create context
	cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &error);
	check_error(error);

	// Create a command queue
	cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, nullptr, &error);
	check_error(error);


	// Load kernel source
	const char* kernelSource = loadKernelFromFile("src/kernel.cl");

	// Create program from source
	cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, &error);
	check_error(error);

	// Build program
	error = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
	check_error(error);

	// Create kernel
	cl_kernel kernel = clCreateKernel(program, "blurAxis", &error);
	check_error(error);


	// Create buffers
	cl_mem d_input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, img_size, img_in, &error);
	check_error(error);
	cl_mem d_temp = clCreateBuffer(context, CL_MEM_READ_WRITE, img_size, nullptr, &error);
	check_error(error);
	cl_mem d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, img_size, nullptr, &error);
	check_error(error);
	cl_mem d_weights = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * weights.size(), weights.data(), &error);
	check_error(error);

	
	// Launch horizontal blur
	int axis = 0;
	clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_temp);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_weights);
	clSetKernelArg(kernel, 3, sizeof(int), &width);
	clSetKernelArg(kernel, 4, sizeof(int), &height);
	clSetKernelArg(kernel, 5, sizeof(int), &axis);

	size_t local_work_size[2] = { 16, 16 };
	size_t global_work_size[2] = { (size_t)width, (size_t)height };
	clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global_work_size, local_work_size, 0, nullptr, nullptr);
	clFlush(queue);
	clFinish(queue);
	

	// Launch vertical blur
	axis = 1;
	clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_temp);	
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_output);
	clSetKernelArg(kernel, 5, sizeof(int), &axis);

	clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global_work_size, local_work_size, 0, nullptr, nullptr);
	clFlush(queue);
	clFinish(queue);

	// Read result
	clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, img_size, img_out, 0, nullptr, nullptr);


	auto end = std::chrono::high_resolution_clock::now();
	// Computation time in milliseconds
	int time = (int)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	printf("Gaussian Blur Parallel : Time %dms\n", time);

	// Write the blurred image into a JPG file
	stbi_write_jpg("images/image_blurred_final.jpg", width, height, 4/*channels*/, img_out, 90 /*quality*/);

	// Release resources
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseMemObject(d_input);
	clReleaseMemObject(d_output);
	clReleaseMemObject(d_weights);
	clReleaseMemObject(d_temp);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
	stbi_image_free(img_in);
	delete[] img_out;
}

int main()
{
	const char* filename = "images/street_night.jpg";
	//gaussian_blur_separate_serial(filename);
	gaussian_blur_separate_parallel(filename);

	return 0;
}