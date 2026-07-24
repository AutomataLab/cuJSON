#include <stdio.h>
#include <iostream>
#include <stdint.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <thread>
#include <x86intrin.h>
#include <string.h>
#include <bitset>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/gather.h>
#include <thrust/extrema.h>
#include <thrust/partition.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <inttypes.h>
#include <thrust/host_vector.h>
#include <device_launch_parameters.h>
#include <vector>
#include <cstring>
#include <cub/cub.cuh>




// #include "./12-GJSON-Class.cuh"

#define         MAXLINELENGTH     1073741824   //4194304 8388608 33554432 67108864 134217728 201326592 268435456 536870912 805306368 1073741824// Max record size
                                              //4MB       8MB     32BM    64MB      128MB    192MB     256MB     512MB     768MB       1GB
#define         BUFSIZE           1073741824   //4194304 8388608 33554432 67108864 134217728 201326592 268435456 536870912 805306368 1073741824

#define BLOCKSIZE 256

#define OPENBRACKET 91
#define CLOSEBRACKET 93
#define OPENBRACE 123
#define CLOSEBRACE 125
#define I 73

#define ROW1 1
#define ROW2 2
#define ROW3 3
#define ROW4 4
#define ROW5 5

#ifndef DEBUG_MODE
    #define DEBUG_MODE 1 
    // Set to 5 for debugging (memory consumption),
    // Set to 4 for debugging (size),
    // Set to 3 for debugging (overall time report), 
    // Set to 2 for debugging (time report), 
    // Set to 1 for debugging (print), 0 for production
#endif

using namespace std;
using namespace std::chrono;

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip> // For formatting

void printGpuMemoryUsage(const std::string& msg = "") {
    size_t free_mem = 0;
    size_t total_mem = 0;

    cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemGetInfo failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    size_t used_mem = total_mem - free_mem;

    std::cout << std::fixed << std::setprecision(2);
    if (!msg.empty()) {
        std::cout << "[" << msg << "] \n";
    }
    std::cout << "GPU Memory Usage: "
              << "Used = " << used_mem / (1024.0 * 1024.0) << " MB, "
              << "Free = " << free_mem / (1024.0 * 1024.0) << " MB, "
              << "Total = " << total_mem / (1024.0 * 1024.0) << " MB" << std::endl;
}



// Struct to check if a given integer is equal to 1.
struct is_one {
    __host__ __device__ // Can be called from both host (CPU) and device (GPU) code.
    bool operator()(const int x) { 
        return (x == 1); // Returns true if x is 1.
    }
};

// Struct to check if a character is an opening brace or bracket.
struct is_opening {
    __host__ __device__ // Can be called from both host (CPU) and device (GPU) code.
    bool operator()(char x) {
        return (x == OPENBRACE) || (x == OPENBRACKET); // Returns true for '{' or '['.
    }
};

// Struct to check if a character is a closing brace or bracket.
struct is_closing {
    __host__ __device__ // Can be called from both host (CPU) and device (GPU) code.
    bool operator()(char x) {
        return (x == CLOSEBRACE) || (x == CLOSEBRACKET); // Returns true for '}' or ']'.
    }
};

// Struct to decrease an integer by 1.
struct decrease {
    __host__ __device__ // Can be called from both host (CPU) and device (GPU) code.
    int operator()(int x) {
        return x - 1; // Decreases the input integer by 1.
    }
};

// Struct to increase an integer by 1.
struct increase {
    __host__ __device__ // Can be called from both host (CPU) and device (GPU) code.
    int operator()(int x) {
        return x + 1; // Increases the input integer by 1.
    }
};





// Inline device function to compute the prefix XOR for a 32-bit integer.
// This function performs XOR-based prefix computations for efficiency.
__device__ __forceinline__
uint32_t prefix_xor(uint32_t x) {
    x ^= (x << 1);   // XOR with left-shifted version by 1 bit.
    x ^= (x << 2);   // XOR with left-shifted version by 2 bits.
    x ^= (x << 4);   // XOR with left-shifted version by 4 bits.
    x ^= (x << 8);   // XOR with left-shifted version by 8 bits.
    x ^= (x << 16);  // XOR with left-shifted version by 16 bits.
    return x;        // Returns the resulting XOR value.
}

// Inline device function to compute the prefix XOR for a 64-bit integer.
// This function performs XOR-based prefix computations for efficiency.
__device__ __forceinline__
uint64_t prefix_xor64(uint64_t x) {
    x ^= (x << 1);   // XOR with left-shifted version by 1 bit.
    x ^= (x << 2);   // XOR with left-shifted version by 2 bits.
    x ^= (x << 4);   // XOR with left-shifted version by 4 bits.
    x ^= (x << 8);   // XOR with left-shifted version by 8 bits.
    x ^= (x << 16);  // XOR with left-shifted version by 16 bits.
    x ^= (x << 32);  // XOR with left-shifted version by 32 bits.
    return x;        // Returns the resulting XOR value.
}




// _______________________Debug__Functions_______________________

// _______________________Device_Functions_______________________
// Converts a single byte (8 bits) into its binary string representation.
// Returns a pointer to a shared memory string containing the binary representation.
// __device__
// const char* byteToBinary(uint8_t byte) {
//     __shared__ char binary[9]; // Shared memory buffer for binary string (ensure no race conditions).
//     binary[8] = '\0'; // Null terminator for the binary string.

//     for (int i = 7; i >= 0; --i) {
//         binary[i] = (byte & 0x01) ? '1' : '0'; // Extract the least significant bit.
//         byte >>= 1; // Shift right to process the next bit.
//     }

//     return binary;
// }

// // Converts a 32-bit unsigned integer into its binary string representation.
// // The output buffer `out` must have at least 33 characters (32 bits + null terminator).
// __device__
// void u32ToBinary(uint32_t num, char* out) {
//     out[32] = '\0'; // Null terminator for the binary string.
//     for (int i = 31; i >= 0; --i) {
//         out[i] = (num & 0x01) ? '1' : '0'; // Extract the least significant bit.
//         num >>= 1; // Shift right to process the next bit.
//     }
// }


// _______________________Host_Functions_for_Debugging_GPU_Data_______________________

// Print the first 100 characters of the XML file.
void printFirst100Chars(const uint8_t* data, size_t length) {
    if (!data || length == 0) {
        std::cerr << "\033[1;34m Warning: No data to print! \033[0m" << std::endl;
        return;
    }

    std::cout << "First 100 characters of XML file:\n";
    for (size_t i = 0; i < 100 && i < length; i++) {
        std::cout << static_cast<char>(data[i]);
    }
    std::cout << std::endl;
}

// Prints a 2D array of 32-bit unsigned integers stored on the GPU.
// Converts each value to its binary representation and outputs row by row.
void print_d32(uint32_t* d_data, int total_padded_32, int rows) {
    uint32_t* h_data = (uint32_t*)malloc(total_padded_32 * rows * sizeof(uint32_t)); // Host buffer.
    if (!h_data) {
        std::cerr << "Failed to allocate host memory!" << std::endl;
        return;
    }

    cudaMemcpy(h_data, d_data, total_padded_32 * rows * sizeof(uint32_t), cudaMemcpyDeviceToHost); // Copy from device to host.

    for (int i = 0; i < total_padded_32 * rows; ++i) {
        uint32_t value = h_data[i];
        for (int j = 0; j < 32; ++j) { // Print each bit of the value.
            std::cout << ((value >> j) & 1);
        }
        std::cout << std::endl;
    }

    free(h_data); // Free host memory.
}

// Prints selected portions of a 2D array of 32-bit unsigned integers stored on the GPU.
int print_d(uint32_t* input_GPU, int length, int rows) {
    uint32_t* input = (uint32_t*)malloc(sizeof(uint32_t) * length * rows); // Host buffer.
    cudaMemcpyAsync(input, input_GPU, sizeof(uint32_t) * length * rows, cudaMemcpyDeviceToHost); // Async copy to host.

    for (long i = 0; i < rows; i++) {
        for (long j = 401; j < 470 && j < length; j++) { // Print a specific range of columns.
            std::bitset<32> y(*(input + j + (i * length))); // Convert to binary using std::bitset.
            if (j == 129) printf("----129----");
            std::cout << y << ' ';
        }
        std::cout << "\n";
    }

    free(input); // Free host memory.
    return 1;
}

// Prints a 2D array of 8-bit unsigned integers stored on the host.
int print8(uint8_t* input, int length, int rows) {
    for (long i = 0; i < rows; i++) {
        for (long j = 0; j < length && j < 200; j++) { // Print up to 200 values per row.
            std::cout << *(input + j + (i * length)) << ' ';
        }
        std::cout << std::endl;
    }
    return 1;
}

// Prints a 2D array of 32-bit integers stored on the host.
int print32(int32_t* input, int length, int rows) {
    for (long i = 0; i < rows; i++) {
        for (long j = 0; j < length && j < 200; j++) { // Print up to 200 values per row.
            std::cout << *(input + j + (i * length)) << ' ';
        }
        std::cout << std::endl;
    }
    return 1;
}

// Template function to print a 2D array of 8-bit integers from the GPU.
// The array is transferred to the host before printing.
template<typename T>
int print8_d(uint8_t* input_GPU, int length, int rows) {
    uint8_t* input = (uint8_t*)malloc(sizeof(uint8_t) * length); // Host buffer.
    cudaMemcpyAsync(input, input_GPU, sizeof(uint8_t) * length, cudaMemcpyDeviceToHost); // Async copy to host.

    for (long i = 0; i < rows; i++) {
        for (long j = 0; j < 300 && j < length; j++) { // Print up to 300 values per row.
            std::cout << (T)*(input + j + (i * length)) << ' ';
        }
        std::cout << std::endl;
    }

    free(input); // Free host memory.
    return 1;
}


void printByteByByte(int32_t* data, int length) {
    for (int i = 0; i < length; ++i) {
        unsigned char* bytePointer = (unsigned char*)&data[i];
        for (int j = 0; j < sizeof(int32_t); ++j) {
            printf("%02x ", bytePointer[j]);
        }
        printf("\n");
    }
}

// Prints a 2D array of 32-bit unsigned integers stored on the GPU.
void print_token_array_as_bytes(const char* label, uint32_t* d_array, size_t length_uint32) {
    size_t length_bytes = length_uint32 * sizeof(uint32_t);
    uint8_t* h_bytes = new uint8_t[length_bytes];

    cudaMemcpy(h_bytes, d_array, length_bytes, cudaMemcpyDeviceToHost);

    std::cout << "=== " << label << " (bytes) ===\n";
    for (size_t i = 0; i < length_bytes && i < 50; i++) {
        printf("%3d ", h_bytes[i]);  // Print as decimal byte
        if ((i + 1) % 8 == 0) std::cout << "\n";
    }
    std::cout << "\n";

    delete[] h_bytes;
}

// Prints the first 100 token indices from a device array of uint32_t.
void print_uint32_indices(const char* label, const uint32_t* d_finalTokens, size_t tokens_count) {
    std::cout << "=== " << label << " (token indices) ===\n";

    // Copy to host
    uint32_t* h_tokens = new uint32_t[tokens_count];
    cudaMemcpy(h_tokens, d_finalTokens, tokens_count * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Print 8 per line
    for (size_t i = 0; i < tokens_count && i < 50; ++i) {
        std::cout << h_tokens[i] << " ";
        if ((i + 1) % 8 == 0) std::cout << "\n";
    }
    std::cout << "\n";

    delete[] h_tokens;
}

// Prints the first 100 token indices and their corresponding character values from the device arrays.
void print_token_info(const char* label, const uint32_t* d_indices, const uint8_t* d_values, size_t count) {
    std::cout << "=== " << label << " (token indices : char) ===\n";

    std::vector<uint32_t> h_indices(count);
    std::vector<uint8_t> h_values(count);

    cudaMemcpy(h_indices.data(), d_indices, count * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_values.data(), d_values, count * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < count && i < 50; ++i) {
        std::cout << h_indices[i] << " : '" << static_cast<char>(h_values[i]) << "'\n";
    }
}

// Prints the byte map of a device array (uint8_t) for debugging purposes.
void print_byte_map(const char* label, const uint8_t* d_array, size_t length) {
    #if defined(DEBUG_MODE) && DEBUG_MODE == 1
        std::cout << "=== " << label << " (byte-by-byte) ===\n";
    
        // Allocate host memory and copy from device
        uint8_t* h_array = new uint8_t[length];
        cudaMemcpy(h_array, d_array, length * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    
        // Print 8 values per line
        for (size_t i = 0; i < length && i < 50; ++i) {
            printf("%3d ", h_array[i]);
            if ((i + 1) % 8 == 0) std::cout << "\n";
        }
        if (length % 8 != 0) std::cout << "\n";
    
        delete[] h_array;
    #endif
}

// Prints the byte signs (1 for opening, -1 for closing) stored on the GPU.
void print_byte_signs(const char* label, const int8_t* d_signs, size_t count) {
    std::cout << "=== " << label << " (byte-by-byte) ===\n";
    
    int8_t* h = new int8_t[count];
    cudaMemcpy(h, d_signs, count * sizeof(int8_t), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < count && i < 50; ++i) {
        printf("%3d ", (int)h[i]);
        if ((i + 1) % 8 == 0) std::cout << "\n";
    }

    if (count % 8 != 0) std::cout << "\n";

    delete[] h;
}

// Prints the depth array stored on the GPU for debugging purposes. 
void print_uint32_array(const char* label, const uint32_t* d_depth, size_t count) {
    std::cout << "=== " << label << " (word-by-word) ===\n";
    
    std::vector<uint32_t> h_depth(count);
    cudaMemcpy(h_depth.data(), d_depth, count * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < count && i < 650; ++i) {
        std::cout << (int) h_depth[i] << "\t"; // Print each depth value
        if ((i + 1) % 8 == 0) std::cout << "\n";
    }
    if (count % 8 != 0) std::cout << "\n";
}

void print_query_output(int8_t* d_output, int num_steps, int tokens_count, int K) {
    int total_size = num_steps * tokens_count;
    std::vector<int8_t> h_output(total_size);

    // Copy from device to host
    cudaMemcpy(h_output.data(), d_output, total_size * sizeof(int8_t), cudaMemcpyDeviceToHost);

    // Print first K rows per step
    std::cout << "=== Output Matrix (d=1, d=2, d=3) ===" << std::endl;
    // std::cout << "Output Matrix (First " << K << " rows per step):\n";
    for (int step = 0; step < num_steps; ++step) {
        std::cout << "Step" << step << ": "<<endl;
        for (int i = 0; i < std::min(K, tokens_count) && i < 650; ++i) {
            int8_t val = h_output[step * tokens_count + i];
            std::cout << (int)val << "\t";  // cast to int for readable output
        }
        std::cout << std::endl;
    }
}

void print_int_device_vector(const thrust::device_vector<int>& d_vec, const std::string& name, int K) {
    std::vector<int> h_vec(d_vec.size());
    cudaMemcpy(h_vec.data(), thrust::raw_pointer_cast(d_vec.data()), h_vec.size() * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "=== " << name << " (First " << K << " elements) ===" << std::endl;
    for (int i = 0; i < std::min(K, (int)h_vec.size()); ++i) {
        std::cout << h_vec[i] << "\t";
    }
    std::cout << std::endl;
}


void print_parsed_query(const vector<vector<string>>& parsed) {
    const vector<string> headers = {
        "Tag Name", "Attr Name", "Attr Cond Op", "Tag Cond Op",
        "Tag Cond Val", "Attr Cond Val", "Query Depth", "Attr Cond Name", "Tag Cond Name", 
        "Index Name", "Index Cond Op", "Index Cond Val"
    };

    size_t num_cols = parsed[0].size();
    cout << "Parsed XPath:\n";
    for (size_t r = 0; r < parsed.size(); ++r) {
        cout << headers[r] << ":\t";
        for (size_t c = 0; c < num_cols; ++c) {
            cout << (parsed[r][c].empty() ? "∅" : parsed[r][c]) << "\t";
        }
        cout << endl;
    }
}



template <typename T>
void print_device_array(const T* d_array, size_t count, const std::string& label) {
    std::vector<T> h_array(count);
    cudaMemcpy(h_array.data(), d_array, count * sizeof(T), cudaMemcpyDeviceToHost);
    std::cout << label << ": " << endl;
    for (size_t i = 0; i < count && i < 650; ++i) {
        std::cout << static_cast<int>(h_array[i]) << "\t";
    }
    std::cout << std::endl;
}

template <typename InputIterator>
void print_thrust_iterator(InputIterator begin, size_t count, const std::string& label) {
    size_t to_copy = std::min(count, static_cast<size_t>(650));
    std::vector<typename thrust::iterator_traits<InputIterator>::value_type> host_vec(to_copy);
    thrust::copy(begin, begin + to_copy, host_vec.begin());

    std::cout << label << ": " << endl;
    for (size_t i = 0; i < to_copy; ++i) {
        std::cout << static_cast<int>(host_vec[i]) << "\t";
    }
    std::cout << std::endl;
}


// ______________________check_CUDA_______________________
// Function to check the status of a CUDA API call and handle errors if any.
// If the CUDA call fails, the function prints the error message and terminates the program.
void checkCuda(cudaError_t result) {
    if (result != cudaSuccess) { // Check if the CUDA call did not succeed.
        // Print the error message associated with the CUDA error.
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        // Exit the program with a non-zero status to indicate an error.
        exit(1);
    }
}

// CUB functions:

// count_ones_cub: Counts the number of ones in a device array using CUB.
// This function uses the CUB library to perform a parallel reduction on the device array.
// It first queries the temporary storage needed for the reduction, then allocates that storage,
// performs the reduction, and finally copies the result back to the host.
// The function returns the count of ones found in the device array.
// The input array is expected to be a device pointer to an array of uint8_t values.
// The length parameter specifies the number of elements in the array.
// The function returns the count of ones found in the device array.
uint32_t count_ones_cub(uint8_t* d_flags, size_t length){
    // 1. temporary‑storage query
    void*  d_temp  = nullptr;
    size_t temp_sz = 0;
    uint32_t* d_result;                      // device scalar result
    cudaMalloc(&d_result, sizeof(uint32_t));

    cub::DeviceReduce::Sum(
        d_temp,     temp_sz,                 // temp buffer ptr & size
        d_flags,    d_result,                // in, out
        length);                             // # elements

    // 2. allocate temp storage and run the real reduction
    cudaMalloc(&d_temp, temp_sz);

    cub::DeviceReduce::Sum(
        d_temp, temp_sz,
        d_flags, d_result,
        length);

    // 3. copy result back to host
    uint32_t h_count = 0;
    cudaMemcpy(&h_count, d_result, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // 4. clean‑up
    cudaFree(d_temp);
    cudaFree(d_result);

    return h_count;
}

uint32_t reduce_cub_int(int8_t* d_flags, size_t length){
    // 1. temporary‑storage query
    void*  d_temp  = nullptr;
    size_t temp_sz = 0;
    uint32_t* d_result;                      // device scalar result
    cudaMalloc(&d_result, sizeof(uint32_t));

    cub::DeviceReduce::Sum(
        d_temp,     temp_sz,                 // temp buffer ptr & size
        d_flags,    d_result,                // in, out
        length);                             // # elements

    // 2. allocate temp storage and run the real reduction
    cudaMalloc(&d_temp, temp_sz);

    cub::DeviceReduce::Sum(
        d_temp, temp_sz,
        d_flags, d_result,
        length);

    // 3. copy result back to host
    uint32_t h_count = 0;
    cudaMemcpy(&h_count, d_result, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // 4. clean‑up
    cudaFree(d_temp);
    cudaFree(d_result);

    return h_count;
}



// inclusive_scan_inplace_cub: Performs an inclusive scan on a device array using CUB.
// This function uses the CUB library to perform an inclusive scan (prefix sum) on the input array.
// The input array is expected to be a device pointer to an array of int8_t values.
// The length parameter specifies the number of elements in the array.
// The function modifies the input array in place, storing the result back in the same array.
// The function does not return any value.
// It allocates temporary storage for the scan operation, performs the scan, and then frees the temporary storage.
// The input array is expected to be a device pointer to an array of int8_t values.
// The length parameter specifies the number of elements in the array.
// The function modifies the input array in place, storing the result back in the same array.
// The function does not return any value.
// It allocates temporary storage for the scan operation, performs the scan, and then frees the temporary storage.
// The function is designed to be efficient and uses CUB's parallel algorithms for the scan operation.
void inclusive_scan_inplace_cub(int8_t* d_data, size_t length) {
    // Allocate temp buffer
    void* d_temp = nullptr;
    size_t temp_bytes = 0;

    // First call: query temp storage
    cub::DeviceScan::InclusiveSum(
        d_temp, temp_bytes,
        d_data, d_data,  // in-place
        length
    );

    // Allocate temp buffer
    cudaMalloc(&d_temp, temp_bytes);

    // Second call: actual inclusive scan
    cub::DeviceScan::InclusiveSum(
        d_temp, temp_bytes,
        d_data, d_data,
        length
    );

    // Free temp storage
    cudaFree(d_temp);
}



// A small helper to count “1”s in a virtual flag stream using CUB
template<class FlagIter>
uint32_t count_virtual_flags_cub(FlagIter flags, size_t length ) {
    // device storage for the result
    uint32_t* d_result = nullptr;
    cudaMalloc(&d_result, sizeof(uint32_t));

    // 1a) query temp storage size
    void*   d_temp   = nullptr;
    size_t  temp_sz  = 0;
    cub::DeviceReduce::Sum(
        d_temp, temp_sz,
        flags,    // could be a transform_iterator
        d_result, // device scalar
        length
    );

    // 1b) allocate temp storage & run real reduction
    cudaMalloc(&d_temp, temp_sz);
    cub::DeviceReduce::Sum(
        d_temp, temp_sz,
        flags,
        d_result,
        length
    );

    // 1c) copy back
    uint32_t h_result = 0;
    cudaMemcpy(&h_result, d_result, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // clean up
    cudaFree(d_temp);
    cudaFree(d_result);

    return h_result;
}


// Copies elements from d_token_indices to d_selected_token_indices where d_output_flag == 1
// This function uses CUB's DeviceSelect::Flagged to perform the scatter operation.
// The input array d_token_indices is expected to be a device pointer to an array of uint32_t values.
// The d_output_flag array is a device pointer to an array of uint8_t values, where 1 indicates the element should be copied.
// The d_selected_token_indices array is a device pointer to an array of uint32_t values, where the selected elements will be copied.
// The tokens_count parameter specifies the number of elements in the input array.
// The function does not return any value.
// It allocates temporary storage for the scatter operation, performs the scatter, and then frees the temporary storage.
// The function is designed to be efficient and uses CUB's parallel algorithms for the scatter operation.

void scatter_cub(
    const uint32_t* d_token_indices,        // input data
    const uint8_t* d_output_flag,           // 0/1 flag for selection
    uint32_t* d_selected_token_indices,     // output buffer (preallocated)
    size_t tokens_count                     // number of input items
) {
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    uint32_t* d_num_selected_out;

    // Allocate temporary output count
    cudaMalloc(&d_num_selected_out, sizeof(uint32_t));

    // Step 1: Query temporary storage size
    cub::DeviceSelect::Flagged(
        d_temp_storage, temp_storage_bytes,
        d_token_indices,            // input values
        d_output_flag,              // stencil
        d_selected_token_indices,   // output
        d_num_selected_out,         // number selected
        tokens_count
    );

    // Step 2: Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Step 3: Run the actual selection
    cub::DeviceSelect::Flagged(
        d_temp_storage, temp_storage_bytes,
        d_token_indices,
        d_output_flag,
        d_selected_token_indices,
        d_num_selected_out,
        tokens_count
    );

    // Optional: get count back (if you want to use it later)
    // uint32_t h_selected_count = 0;
    // cudaMemcpy(&h_selected_count, d_num_selected_out, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_temp_storage);
    cudaFree(d_num_selected_out);
}


template<class FlagIter>
void scatter_virtual_flag_cub(
    const uint32_t*        d_token_indices,         // input indices
    size_t                 tokens_count,           // number of tokens
    FlagIter               flags,                  // transform_iterator over [0..tokens_count)
    uint32_t*              d_selected_tokens,      // OUT: compacted indices
    uint32_t*              d_selected_count       // OUT: device scalar count
) {
    // 2a) query temp storage
    void*  d_temp  = nullptr;
    size_t temp_sz = 0;
    auto  index_begin = thrust::make_counting_iterator<uint32_t>(0);

    cub::DeviceSelect::Flagged(
       d_temp, temp_sz,
       index_begin,     // the “input items” (we really only care about the index)
       flags,           // the lazy flag stream
       d_selected_tokens,
       d_selected_count,
       tokens_count
    );

    // 2b) allocate & run
    cudaMalloc(&d_temp, temp_sz);
    cub::DeviceSelect::Flagged(
       d_temp, temp_sz,
       index_begin,
       flags,
       d_selected_tokens,
       d_selected_count,
       tokens_count
    );
    cudaFree(d_temp);
}



