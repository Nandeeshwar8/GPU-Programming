
#include <iostream>
#include <cuda_runtime.h>

int getCoresPerSM(int major, int minor)
{
    // Defines cores per SM based on architecture generation
    switch (major)
    {
        case 2: // Fermi
            return (minor == 1) ? 48 : 32;

        case 3: // Kepler
            return 192;

        case 5: // Maxwell
            return 128;

        case 6: // Pascal
            if (minor == 1 || minor == 2)
                return 128;
            if (minor == 0)
                return 64;
            return 128;

        case 7: // Volta / Turing
            return 64;

        case 8: // Ampere / Ada
            if (minor == 0)
                return 64;
            if (minor == 6 || minor == 9)
                return 128;
            return 64;

        case 9: // Hopper / Blackwell
            return 128;

        default:
            return 128;
    }
}

int main()
{
    int deviceCount = 0;

    
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess)
    {
        std::cerr << "CUDA Error: "
                  << cudaGetErrorString(error)
                  << std::endl;
        return 1;
    }

    std::cout << "Found " << deviceCount
              << " CUDA device(s).\n"
              << std::endl;

   
    for (int i = 0; i < deviceCount; ++i)
    {
        cudaDeviceProp prop;

        
        cudaGetDeviceProperties(&prop, i);

        int coresPerSM = getCoresPerSM(prop.major, prop.minor);
        int totalCores = prop.multiProcessorCount * coresPerSM;

      

        
        std::cout << "Device Name:                  "
                  << prop.name << std::endl;

        std::cout << "Compute Capability:           "
                  << prop.major << "."
                  << prop.minor << std::endl;

        std::cout << "Total Global Memory:          "
                  << prop.totalGlobalMem / (1024 * 1024)
                  << " MB" << std::endl;

        std::cout << "Streaming Multiprocessors:    "
                  << prop.multiProcessorCount << std::endl;

        std::cout << "Cores Per SM:                 "
                  << coresPerSM << std::endl;

        std::cout << "Total Cores:                  "
                  << totalCores << std::endl;

        std::cout << "Max Threads Per Block:        "
                  << prop.maxThreadsPerBlock << std::endl;

        std::cout << "Shared Memory Per Block:      "
                  << prop.sharedMemPerBlock / 1024
                  << " KB" << std::endl;

        std::cout << "Warp Size:                    "
                  << prop.warpSize << std::endl;


        

        
        std::cout << "Clock Rate:                   "
                  << prop.clockRate / 1000
                  << " MHz" << std::endl;

        
        std::cout << "Memory Clock Rate:            "
                  << prop.memoryClockRate / 1000
                  << " MHz" << std::endl;

        
        std::cout << "Memory Bus Width:             "
                  << prop.memoryBusWidth
                  << " bits" << std::endl;

        
        std::cout << "L2 Cache Size:                "
                  << prop.l2CacheSize / 1024
                  << " KB" << std::endl;

        
        std::cout << "Registers Per Block:          "
                  << prop.regsPerBlock << std::endl;

        
        std::cout << "Max Threads Per SM:           "
                  << prop.maxThreadsPerMultiProcessor
                  << std::endl;

       
        std::cout << "Constant Memory:              "
                  << prop.totalConstMem / 1024
                  << " KB" << std::endl;

        
        std::cout << "Max Grid Dimensions:          "
                  << prop.maxGridSize[0] << " x "
                  << prop.maxGridSize[1] << " x "
                  << prop.maxGridSize[2]
                  << std::endl;

        
        std::cout << "Max Block Dimensions:         "
                  << prop.maxThreadsDim[0] << " x "
                  << prop.maxThreadsDim[1] << " x "
                  << prop.maxThreadsDim[2]
                  << std::endl;

      
        std::cout << "Concurrent Kernels:           "
                  << (prop.concurrentKernels ? "Yes" : "No")
                  << std::endl;


        
        std::cout << "\n\n";

        std::cout << "Texture Alignment:            "
                  << prop.textureAlignment
                  << " bytes" << std::endl;

        std::cout << "Memory Pitch:                 "
                  << prop.memPitch / (1024 * 1024)
                  << " MB" << std::endl;

        std::cout << "ECC Enabled:                  "
                  << (prop.ECCEnabled ? "Yes" : "No")
                  << std::endl;

        std::cout << "Unified Addressing:            "
                  << (prop.unifiedAddressing ? "Yes" : "No")
                  << std::endl;

        std::cout << "Can Map Host Memory:           "
                  << (prop.canMapHostMemory ? "Yes" : "No")
                  << std::endl;

        std::cout << "\n\n";
    }

    return 0;
}