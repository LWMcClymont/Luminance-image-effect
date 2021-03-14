// Sampler
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
								CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

// Luminance values stored in vector4 for dot product
__constant float4 luminanceVals = (float4)(0.299, 0.587f, 0.114f, 0);

__kernel void task (read_only image2d_t inputImage, write_only image2d_t outputImage)
{
	// Get coordinates
	int2 coord = (int2)(get_global_id(0), get_global_id(1));

	// Get pixel
	float4 pixel = read_imagef (inputImage, sampler, coord);

	// Perform dot product and write pixel to output
	write_imagef (outputImage, coord, dot(pixel, luminanceVals));

}