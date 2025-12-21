#include <SDL3/SDL.h>
#include <SDL3/SDL_gpu.h>

#include <vector>

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>







// Helper function to create a shader
SDL_GPUShader* createShader(SDL_GPUDevice* device, const char* code, SDL_GPUShaderStage stage, const char* entrypoint, int uni  );


// Function to create a basic pipeline - note: returning by value is unusual
// and likely problematic for SDL objects, but implementing as requested
SDL_GPUGraphicsPipeline* createBasicPipeline(SDL_GPUDevice* device);
// Create multiple pipelines with different configurations
std::vector<SDL_GPUGraphicsPipeline*> createPipelines(SDL_GPUDevice* device);
// Utility to cleanup pipelines
void releasePipelines(SDL_GPUDevice* device, std::vector<SDL_GPUGraphicsPipeline*>& pipelines);