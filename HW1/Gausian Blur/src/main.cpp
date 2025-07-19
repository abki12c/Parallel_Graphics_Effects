#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <chrono>
#include <thread>
#include <mutex>
#include <barrier>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define THREADS_NUMBER 2

const int KERNEL_RADIUS = 8;
const float sigma = 3.f;

std::barrier bar(4);

unsigned char blur(int x, int y, int channel, unsigned char* input, int width, int height)
{
	float sum_weight = 0.0f;
	float ret = 0.f;

	for (int offset_y = -KERNEL_RADIUS; offset_y <= KERNEL_RADIUS; offset_y++)
	{
		for (int offset_x = -KERNEL_RADIUS; offset_x <= KERNEL_RADIUS; offset_x++)
		{
			int pixel_y = std::max(std::min(y + offset_y, height - 1), 0);
			int pixel_x = std::max(std::min(x + offset_x, width - 1), 0);
			int pixel = pixel_y * width + pixel_x;

			float weight = std::exp(-(offset_x * offset_x + offset_y * offset_y) / (2.f * sigma * sigma));

			ret += weight * input[4 * pixel + channel];
			sum_weight += weight;
		}
	}
	ret /= sum_weight;

	return (unsigned char)std::max(std::min(ret, 255.f), 0.f);
}

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

void gaussian_blur_serial(const char* filename)
{
	int width = 0;
	int height = 0;
	int img_orig_channels = 4;
	// Load an image into an array of unsigned chars that is the size of [width * height * number of channels]. The channels are the Red, Green, Blue and Alpha channels of the image.
	unsigned char* img_in = stbi_load(filename, &width, &height, &img_orig_channels /*image file channels*/, 4 /*requested channels*/);
	if (img_in == nullptr)
	{
		printf("Could not load %s\n", filename);
		return;
	}

	unsigned char* img_out = new unsigned char[width * height * 4];

	// Timer to measure performance
	auto start = std::chrono::high_resolution_clock::now();

	// Perform Gaussian Blur to each pixel
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			int pixel = y * width + x;
			for (int channel = 0; channel < 4; channel++)
			{
				img_out[4 * pixel + channel] = blur(x, y, channel, img_in, width, height);
			}
		}
	}

	// Timer to measure performance
	auto end = std::chrono::high_resolution_clock::now();
	// Computation time in milliseconds
	int time = (int)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	printf("Gaussian Blur - Serial: Time %dms\n", time);

	// Write the blurred image into a JPG file
	stbi_write_jpg("images/blurred_image_serial.jpg", width, height, 4, img_out, 90 /*quality*/);

	stbi_image_free(img_in);
	delete[] img_out;
}


void calculate_pixels(int y_start, int y_end, unsigned char* img_in, int width, int height, unsigned char* img_out)
{
	for (int y = y_start; y < y_end; y++)
	{
		for (int x = 0; x < width; x++)
		{
			int pixel = y * width + x;
			for (int channel = 0; channel < 4; channel++)
			{
				img_out[4 * pixel + channel] = blur(x, y, channel, img_in, width, height);
			}
		}
	}
}

void gaussian_blur_parallel(const char* filename) 
{
	const int threads_number = THREADS_NUMBER;
	int width = 0;
	int height = 0;
	int img_orig_channels = 4;
	// Load an image into an array of unsigned chars that is the size of [width * height * number of channels]. The channels are the Red, Green, Blue and Alpha channels of the image.
	unsigned char* img_in = stbi_load(filename, &width, &height, &img_orig_channels /*image file channels*/, 4 /*requested channels*/);
	if (img_in == nullptr)
	{
		printf("Could not load %s\n", filename);
		return;
	}

	int chunk_size = height / threads_number;

	unsigned char* img_out = new unsigned char[width * height * 4];

	// Timer to measure performance
	auto start = std::chrono::high_resolution_clock::now();

	std::thread threads[threads_number];
	for (int i = 0; i < threads_number;i++) {
		int y_start = i * chunk_size;
		int y_end = (i == threads_number - 1) ? height : y_start + chunk_size;

		// Perform Gaussian Blur to the pixels of a number of rows
		threads[i] = std::thread(calculate_pixels,y_start, y_end, img_in, width, height, img_out);
	}

	// Wait for all threads to finish
	for (int i = 0; i < threads_number;i++) {
		threads[i].join();
	}
	

	// Timer to measure performance
	auto end = std::chrono::high_resolution_clock::now();
	// Computation time in milliseconds
	int time = (int)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	printf("Gaussian Blur - Parallel: Time %dms\n", time);

	// Write the blurred image into a JPG file
	stbi_write_jpg("images/blurred_image_parallel.jpg", width, height, 4, img_out, 90 /*quality*/);


	// free resources
	stbi_image_free(img_in);
	delete[] img_out;
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

void worker(unsigned char* img_in, int width, int height, unsigned char max_channel_value[], unsigned char* img_normalized, unsigned char* img_horizontal_blur, unsigned char* img_out, int channel) {
	unsigned char max_value = 0;

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			int pixel = y * width + x;
			
			if (img_in[4 * pixel + channel] > max_value) {
				max_value = img_in[4 * pixel + channel];
			}
		}
	}

	max_channel_value[channel] = max_value;

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			int pixel = y * width + x;
			
			img_normalized[4 * pixel + channel] = 255 * img_in[4 * pixel + channel] / max_channel_value[channel];
		}
	}

	// wait for normalized image to be completed
	bar.arrive_and_wait();


	// Horizontal blur on normalized image 
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			int pixel = y * width + x;
			img_horizontal_blur[4 * pixel + channel] = blurAxis(x, y, channel, 0, img_normalized, width, height);
		}
	}

	// Wait for horizontal blur to complete
	bar.arrive_and_wait();

	// Vertical blur on horizontally blurred image
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			int pixel = y * width + x;
			img_out[4 * pixel + channel] = blurAxis(x, y, channel, 1, img_horizontal_blur, width, height);
		}
	}


}


void gaussian_blur_separate_parallel(const char* filename)
{
	unsigned char max_channel_value[4] = {0,0,0,0};
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
	unsigned char* img_normalized = new unsigned char[width * height * 4];

	// Timer to measure performance
	auto start = std::chrono::high_resolution_clock::now();

	std::thread thread1 = std::thread(worker, img_in, width, height, max_channel_value, img_normalized, img_horizontal_blur, img_out, 0);
	std::thread thread2 = std::thread(worker, img_in, width, height, max_channel_value, img_normalized, img_horizontal_blur, img_out, 1);
	std::thread thread3 = std::thread(worker, img_in, width, height, max_channel_value, img_normalized, img_horizontal_blur, img_out, 2);
	std::thread thread4 = std::thread(worker, img_in, width, height, max_channel_value, img_normalized, img_horizontal_blur, img_out, 3);

	

	thread1.join();
	thread2.join();
	thread3.join();
	thread4.join();

	// Timer to measure performance
	auto end = std::chrono::high_resolution_clock::now();
	// Computation time in milliseconds
	int time = (int)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	printf("Gaussian Blur Separate - Parallel: Time %dms\n", time);

	for (int i = 0; i < 4;i++) {
		std::cout << "channel " << i + 1 << " max value: " << static_cast<int>(max_channel_value[i]) << std::endl;
	}
	

	// Write the blurred image into a JPG file
	stbi_write_jpg("images/image_normalized.jpg", width, height, 4, img_normalized, 90);
	stbi_write_jpg("images/image_blurred_horizontal.jpg", width, height, 4, img_horizontal_blur, 90);
	stbi_write_jpg("images/image_blurred_final.jpg", width, height, 4, img_out, 90);

	stbi_image_free(img_in);
	delete[] img_horizontal_blur;
	delete[] img_out;
	delete[] img_normalized;
}



int main()
{
	const char* filename = "images/garden.jpg";
	gaussian_blur_serial(filename);
	gaussian_blur_parallel(filename);

	
	
	const char* filename2 = "images/street_night.jpg";
	gaussian_blur_separate_serial(filename2);
	gaussian_blur_separate_parallel(filename2);

	return 0;
}