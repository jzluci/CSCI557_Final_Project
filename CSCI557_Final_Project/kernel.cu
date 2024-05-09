
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <string>
#include <cassert>
#include <chrono>

#include "stb_image.h"
#include "stb_image_write.h"

struct Pixel
{
    unsigned char r, g, b, a;
};

void ConvertImageToGreyCpu(unsigned char* imageRGBA, int width, int height)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            Pixel* ptrPixel = (Pixel*)&imageRGBA[y * width * 4 + 4 * x];
            unsigned char pixelValue = (unsigned char)(ptrPixel->r * 0.2126f + ptrPixel->g * 0.7152f + ptrPixel->b * 0.0722f);
            ptrPixel->r = pixelValue;
            ptrPixel->g = pixelValue;
            ptrPixel->b = pixelValue;
            ptrPixel->a = 255;
        }
    }
}

__global__ void ConvertImageToGreyGPU(unsigned char* imageRGBA)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t idx = y * blockDim.x * gridDim.x + x;

    Pixel* ptrPixel = (Pixel*)&imageRGBA[idx * 4];
    unsigned char pixelValue = (unsigned char)
        (ptrPixel->r * 0.2126f + ptrPixel->g * 0.7152f + ptrPixel->b * 0.0722f);
    ptrPixel->r = pixelValue;
    ptrPixel->g = pixelValue;
    ptrPixel->b = pixelValue;
    ptrPixel->a = 255;

}

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cout << "Usage: CSCI_557_Final_Project <filename>" << std::endl;
        return -1;
    }

    int width, height, componentCount;
    unsigned char* imageData = stbi_load(argv[1], &width, &height, &componentCount, 4);
    if (!imageData) {
        std::cout << "Failed to open \"" << argv[1] << "\"";
        return -1;
    }

    if (width % 32 || height % 32) {
        std::cout << "Width and/or height not divisible by 32";
        return -1;
    }

    // CPU timing and processing
    auto startCpu = std::chrono::high_resolution_clock::now();
    ConvertImageToGreyCpu(imageData, width, height);
    auto endCpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> durationCpu = endCpu - startCpu;
    std::cout << "CPU processing time: " << durationCpu.count() << " ms" << std::endl;

    unsigned char* ptrImageDataGpu = nullptr;
    assert(cudaMalloc(&ptrImageDataGpu, width * height * 4) == cudaSuccess);

    // GPU timing and processing including memory transfers
    cudaEvent_t startGpu, stopGpu;
    cudaEventCreate(&startGpu);
    cudaEventCreate(&stopGpu);
    cudaEventRecord(startGpu);

    assert(cudaMemcpy(ptrImageDataGpu, imageData, width * height * 4, cudaMemcpyHostToDevice) == cudaSuccess);

    dim3 blockSize(32, 32);
    dim3 gridSize(width / blockSize.x, height / blockSize.y);
    ConvertImageToGreyGPU << <gridSize, blockSize >> > (ptrImageDataGpu);

    assert(cudaMemcpy(imageData, ptrImageDataGpu, width * height * 4, cudaMemcpyDeviceToHost) == cudaSuccess);

    cudaEventRecord(stopGpu);
    cudaEventSynchronize(stopGpu);
    float gpuTime = 0;
    cudaEventElapsedTime(&gpuTime, startGpu, stopGpu);
    std::cout << "-GPU processing time: " << gpuTime << " ms" << std::endl;
    cudaEventDestroy(startGpu);
    cudaEventDestroy(stopGpu);

    std::string fileNameOut = argv[1];
    fileNameOut = fileNameOut.substr(0, fileNameOut.find_last_of('.')) + "_grey.png";
    stbi_write_png(fileNameOut.c_str(), width, height, 4, imageData, 4 * width);

    cudaFree(ptrImageDataGpu);
    stbi_image_free(imageData);
    return 0;
}