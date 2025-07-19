const int KERNEL_RADIUS = 8;

__kernel void blurAxis(
    __global const uchar* input,
    __global uchar* output,
    __constant  float* weights,
    const int width,
    const int height,
    const int axis)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height)
        return;

    int pixel_index = y * width + x;

    for (int channel = 0; channel < 4; channel++)
    {
        float ret = 0.0f;
        float sum_weight = 0.0f;

        for (int offset = -KERNEL_RADIUS; offset <= KERNEL_RADIUS; offset++)
        {
            int offset_x = axis == 0 ? offset : 0;
            int offset_y = axis == 1 ? offset : 0;

            int pixel_y = clamp(y + offset_y, 0, height - 1);
		    int pixel_x = clamp(x + offset_x, 0, width - 1);
            int pixel = pixel_y * width + pixel_x;

            float weight = weights[offset + KERNEL_RADIUS];
            ret += weight * input[4 * pixel + channel];
            sum_weight += weight;
        }

        ret /= sum_weight;
        output[4 * pixel_index + channel] = (uchar)(clamp(ret, 0.0f, 255.0f));

    }
}