#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

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

	unsigned char* img_horizontal_blur = new unsigned char[width * height * 4];
	unsigned char* img_out = new unsigned char[width * height * 4];

	// Timer to measure performance
	auto start = std::chrono::high_resolution_clock::now();

	// Horizontal Blur
	int y, x, pixel, channel;
	#pragma omp parallel for schedule(dynamic, 1) private(x, pixel, channel)
	for (y = 0; y < height; y++)
	{
		for (x = 0; x < width; x++)
		{
			pixel = y * width + x;
			for (channel = 0; channel < 4; channel++)
			{
				img_horizontal_blur[4 * pixel + channel] = blurAxis(x, y, channel, 0, img_in, width, height);
			}
		}
	}

	// Vertical Blur
	#pragma omp parallel for schedule(dynamic, 1) private(x, pixel, channel)
	for (y = 0; y < height; y++)
	{
		for (x = 0; x < width; x++)
		{
			pixel = y * width + x;
			for (channel = 0; channel < 4; channel++)
			{
				img_out[4 * pixel + channel] = blurAxis(x, y, channel, 1, img_horizontal_blur, width, height);
			}
		}
	}

	// Timer to measure performance
	auto end = std::chrono::high_resolution_clock::now();
	// Computation time in milliseconds
	int time = (int)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	printf("Gaussian Blur Separate - Parallel: Time %dms\n", time);

	// Write the blurred image into a JPG file
	stbi_write_jpg("images/blurred_image_parallel.jpg", width, height, 4/*channels*/, img_out, 90 /*quality*/);

	stbi_image_free(img_in);
	delete[] img_horizontal_blur;
	delete[] img_out;
}

void bloom_parallel(const char* filename)
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
	unsigned char* img_final = new unsigned char[width * height * 4];
	unsigned char* blurred_mask = new unsigned char[width * height * 4];
	unsigned char* bloom_mask = new unsigned char[width * height * 4];
	unsigned char* luminance = new unsigned char[width * height];
	unsigned char max_luminance = 0;

	// Timer to measure performance
	auto start = std::chrono::high_resolution_clock::now();


	#pragma omp parallel
	{
		// calculate max luminance of all pixels
		int y, x, pixel, channel;
		unsigned char local_max_luminance = 0;
		#pragma omp for schedule(dynamic, 1) private(x, pixel)
		for (y = 0; y < height; y++)
		{
			for (x = 0; x < width; x++)
			{
				pixel = y * width + x;
				luminance[pixel] = (img_in[4 * pixel] + img_in[4 * pixel + 1] + img_in[4 * pixel + 2]) / 3;

				if (luminance[pixel] > local_max_luminance) {
					local_max_luminance = luminance[pixel];
				}
			}
		}

		#pragma omp critical
		{
			if (local_max_luminance > max_luminance) {
				max_luminance = local_max_luminance;
			}
		}

		// create bloom_mask image
		#pragma omp for schedule(dynamic, 1) private(x, pixel, channel)
		for (y = 0; y < height; y++)
		{
			for (x = 0; x < width; x++)
			{
				pixel = y * width + x;
				if (luminance[pixel] > 0.9f * max_luminance) {
					for (channel = 0; channel < 4; channel++)
					{
						bloom_mask[4 * pixel + channel] = img_in[4 * pixel + channel];
					}
				}else {
					bloom_mask[4 * pixel] = 0;
					bloom_mask[4 * pixel + 1] = 0;
					bloom_mask[4 * pixel + 2] = 0;
					bloom_mask[4 * pixel + 3] = 0;
				}
			}
		}

		// Horizontal Blur
		#pragma omp for schedule(dynamic, 1) private(x, pixel, channel)
		for (y = 0; y < height; y++)
		{
			for (x = 0; x < width; x++)
			{
				pixel = y * width + x;
				for (channel = 0; channel < 4; channel++)
				{
					img_horizontal_blur[4 * pixel + channel] = blurAxis(x, y, channel, 0, bloom_mask, width, height);
				}
			}
		}

		// Vertical Blur
		#pragma omp for schedule(dynamic, 1) private(x, pixel, channel)
		for (y = 0; y < height; y++)
		{
			for (x = 0; x < width; x++)
			{
				pixel = y * width + x;
				for (channel = 0; channel < 4; channel++)
				{
					blurred_mask[4 * pixel + channel] = blurAxis(x, y, channel, 1, img_horizontal_blur, width, height);
				}
			}
		}

		int sum;
		#pragma omp for schedule(dynamic, 1) private(x, pixel, channel, sum)
		for (y = 0; y < height; y++)
		{
			for (x = 0; x < width; x++)
			{
				pixel = y * width + x;
				for (channel = 0; channel < 4; channel++)
				{
					sum = img_in[4 * pixel + channel] + blurred_mask[4 * pixel + channel];
					if (sum > 255) sum = 255;
					img_final[4 * pixel + channel] = (unsigned char)sum;
				}
			}
		}

	}
	

	// Timer to measure performance
	auto end = std::chrono::high_resolution_clock::now();
	// Computation time in milliseconds
	int time = (int)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	printf("Maximum Pixel Luminance: %d\n", max_luminance);
	printf("Bloom - Parallel: Time %dms\n", time);

	// Write the blurred image into a JPG file
	stbi_write_jpg("images/bloom_blurred.jpg", width, height, 4/*channels*/, blurred_mask, 90 /*quality*/);
	stbi_write_jpg("images/bloom_final.jpg", width, height, 4/*channels*/, img_final, 90 /*quality*/);

	stbi_image_free(img_in);
	delete[] img_horizontal_blur;
	delete[] blurred_mask;
	delete[] bloom_mask;
	delete[] luminance;
	delete[] img_final;
}

int main()
{
	const char* filename = "images/street_night.jpg";
	gaussian_blur_separate_serial(filename);
	gaussian_blur_separate_parallel(filename);
	
	bloom_parallel(filename);

	return 0;
}