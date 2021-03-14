#define CL_USE_DEPRECATED_OPENCL_2_0_APIS	// using OpenCL 1.2, some functions deprecated in OpenCL 2.0
#define __CL_ENABLE_EXCEPTIONS				// enable OpenCL exemptions

// C++ standard library and STL headers
#include <iostream>
#include <vector>
#include <fstream>

// OpenCL header
#include <CL/cl.hpp>

// Helper files from tutorials
#include "common.h"
#include "bmpfuncs.h"

int main()
{
	cl::Platform platform;			// device's platform
	cl::Device device;				// device used
	cl::Context context;			// context for the device
	cl::Program program;			// OpenCL program object
	cl::Kernel kernel;				// a single kernel object
	cl::CommandQueue queue;			// commandqueue for a context and device

	// Input and output objects
	unsigned char* inputImage;
	unsigned char* outputImage;

	// Image info
	int imageWidth, imageHeight, imageSize;

	cl::ImageFormat imageFormat;
	cl::Image2D inputBuffer, outputBuffer;

	try
	{
		// Select the Device
		if (!select_one_device(&platform, &device))
		{
			quit_program("Device not selected.");
		}

		// Create the context using selected device
		context = cl::Context(device);

		// Build the program
		if (!build_program(&program, &context, "task.cl"))
		{
			quit_program("OpenCL program build error.");
		}

		// Create the kernel 
		kernel = cl::Kernel(program, "task");

		// Create the commandqueue
		queue = cl::CommandQueue(context, device);

		// Read the input file
		inputImage = read_BMP_RGB_to_RGBA("peppers.bmp", &imageWidth, &imageHeight);

		// Allocate memory for the outputs
		imageSize = imageWidth * imageHeight * 4;
		outputImage = new unsigned char[imageSize];


		// Set the image format
		imageFormat = cl::ImageFormat(CL_RGBA, CL_UNORM_INT8); // CL_UNORM_INT8 = 0.0-1.0

		// Create the input image buffer
		inputBuffer = cl::Image2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imageFormat, imageWidth, imageHeight, 0, (void*)inputImage);

		// Create the output image buffers
		outputBuffer = cl::Image2D(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, imageFormat, imageWidth, imageHeight, 0, (void*)outputImage);

		// Set kernal arguments
		kernel.setArg(0, inputBuffer);
		kernel.setArg(1, outputBuffer);

		// Enqueue the kernel
		cl::NDRange offset(0, 0);
		cl::NDRange globalSize(imageWidth, imageHeight);

		queue.enqueueNDRangeKernel(kernel, offset, globalSize);

		std::cout << "Kernel enqueued." << std::endl;
		std::cout << "--------------------" << std::endl;

		// Read images from device to host
		cl::size_t<3> origin, region;
		origin[0] = origin[1] = origin[2] = 0;
		region[0] = imageWidth;
		region[1] = imageHeight;
		region[2] = 1;

		queue.enqueueReadImage(outputBuffer, CL_TRUE, origin, region, 0, 0, outputImage);

		// Write output to file
		write_BMP_RGBA_to_RGB("Task.bmp", outputImage, imageWidth, imageHeight);

		std::cout << "Completed!" << std::endl;

		// Deallocate memory
		free(inputImage);
		free(outputImage);

	}
	catch (cl::Error e) {
		// call function to handle errors
		handle_error(e);
	}

	std::cout << "\npress a key to quit...";
	std::cin.ignore();

	return 0;
}