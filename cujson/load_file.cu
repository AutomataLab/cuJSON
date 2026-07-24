#include "load_file.h"
#include <fstream>
#include <sstream>
#include <iostream>


cuJSONInput loadJSON(const std::string& filePath) { 
    // ______________________LOAD_FILE_____________________________
    std::ifstream file(filePath, std::ios::binary | std::ios::ate);                     // Open in binary mode, seek to end
    if (!file) {                                                                        // unable to open file
        std::cerr << "\033[1;31m Error: Unable to open file: \033[0m \n" << filePath << std::endl;
        return {nullptr, 0};                                                            // Return nullptr and size 0
    }
    
    size_t fileSize = file.tellg();                                                     // Get file size
    file.seekg(0, std::ios::beg);                                                       // Seek back to start


    // allocate pinned memory (Host Memory)
    uint8_t* h_buffer;                                                                  // the place that we store it    
    cudaHostAlloc((void**)&h_buffer, fileSize * sizeof(uint8_t), cudaHostAllocDefault); // allocate pinned memory

    if (!h_buffer) {                                                                    // Unable to allocate pinned memory!
        std::cerr << "\033[1;31m Error: Unable to allocate pinned memory! \033[0m \n" << std::endl;
        file.close();           
        return {nullptr, 0};                                                            // Return nullptr and size 0
    }


    // Read file content into buffer
    file.read(reinterpret_cast<char*>(h_buffer), fileSize);                             // copy from memory to host
    file.close();                                   


    return {h_buffer, fileSize}; // Return the buffer and its size
}

cuJSONLinesInput loadJSONLines_chunkCount(const std::string& filePath, size_t chunkCount = 1) { 
    cuJSONLinesInput input; 
    input.data = nullptr;
    input.size = 0;
    input.chunkCount = chunkCount;

    // ______________________LOAD_FILE_____________________________
    std::ifstream file(filePath, std::ios::binary | std::ios::ate);                     // Open in binary mode, seek to end
    if (!file) {                                                                        // unable to open file
        std::cerr << "\033[1;31m Error: Unable to open file: \033[0m \n" << filePath << std::endl;
        return input;                                                                   // Return nullptr and size 0
    }
    
    size_t fileSize = file.tellg();                                                     // Get file size
    file.seekg(0, std::ios::beg);                                                       // Seek back to start


    // allocate pinned memory (Host Memory)
    uint8_t* h_buffer;                                                                  // the place that we store it    
    cudaHostAlloc((void**)&h_buffer, fileSize * sizeof(uint8_t), cudaHostAllocDefault); // allocate pinned memory

    if (!h_buffer) {                                                                    // Unable to allocate pinned memory!
        std::cerr << "\033[1;31m Error: Unable to allocate pinned memory! \033[0m \n" << std::endl;
        file.close();           
        return input;                                                                   // Return nullptr and size 0
    }


    // Read file content into buffer
    file.read(reinterpret_cast<char*>(h_buffer), fileSize);                             // copy from memory to host
    file.close();   
    

    // Find line offsets
    std::vector<size_t> line_offsets;
    line_offsets.push_back(0);  // Start of first line
    for (size_t i = 0; i < fileSize; ++i) {
        if (h_buffer[i] == '\n') {
            line_offsets.push_back(i + 1);  // Start of next line
        }
    }

    // If last line doesn’t end with \n, manually add EOF as offset
    if (line_offsets.back() < fileSize) line_offsets.push_back(fileSize);

    size_t total_lines = line_offsets.size() - 1;
    size_t lines_per_chunk = total_lines / chunkCount;
    size_t extra_lines = total_lines % chunkCount;

    input.data = h_buffer;
    input.size = fileSize;
    input.chunks.reserve(chunkCount);
    input.chunksSize.reserve(chunkCount);

    size_t line_idx = 0;
    for (size_t i = 0; i < chunkCount; ++i) {
        size_t lines_in_this_chunk = lines_per_chunk + (i < extra_lines ? 1 : 0);

        size_t start_offset = line_offsets[line_idx];
        size_t end_offset = line_offsets[line_idx + lines_in_this_chunk];

        input.chunks.push_back(h_buffer + start_offset);
        input.chunksSize.push_back(end_offset - start_offset);

        line_idx += lines_in_this_chunk;
    }

    
    return input; // Return the buffer and its size
}

cuJSONLinesInput loadJSONLines_chunkSizeBytes(const std::string& filePath, size_t chunkSizeBytes) {
    cuJSONLinesInput input;
    input.data = nullptr;
    input.size = 0;

    // ----------------- LOAD FILE -----------------
    std::ifstream file(filePath, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "\033[1;31m Error: Unable to open file: \033[0m \n" << filePath << std::endl;
        return input;
    }

    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // Allocate pinned memory
    uint8_t* h_buffer;
    cudaHostAlloc((void**)&h_buffer, fileSize * sizeof(uint8_t), cudaHostAllocDefault);
    if (!h_buffer) {
        std::cerr << "\033[1;31m Error: Unable to allocate pinned memory! \033[0m \n" << std::endl;
        file.close();
        return input;
    }

    // Read file
    file.read(reinterpret_cast<char*>(h_buffer), fileSize);
    file.close();

    // ----------------- FIND LINE OFFSETS -----------------
    std::vector<size_t> line_offsets;
    line_offsets.push_back(0);  // First line

    for (size_t i = 0; i < fileSize; ++i) {
        if (h_buffer[i] == '\n') {
            line_offsets.push_back(i + 1);  // Start of next line
        }
    }
    if (line_offsets.back() < fileSize) {
        line_offsets.push_back(fileSize);
    }

    input.data = h_buffer;
    input.size = fileSize;

    size_t current_chunk_start = 0;
    size_t current_offset = 0;
    size_t chunk_count = 0;
    for (size_t i = 1; i < line_offsets.size(); ++i) {
        size_t line_start = line_offsets[i - 1];
        size_t line_end = line_offsets[i];

        // If adding this line would exceed the chunk size, finalize current chunk
        if ((line_end - current_chunk_start) > chunkSizeBytes) {
            chunk_count++;
            input.chunks.push_back(h_buffer + current_chunk_start);
            input.chunksSize.push_back(current_offset - current_chunk_start);
            current_chunk_start = line_start;
        }

        current_offset = line_end;
    }

    // Add last chunk
    if (current_chunk_start < fileSize) {
        input.chunks.push_back(h_buffer + current_chunk_start);
        input.chunksSize.push_back(fileSize - current_chunk_start);
        chunk_count++;
    }

    input.chunkCount = chunk_count;
    return input;
}

cuJSONLinesInput loadJSONLines_chunkSizeMegaBytes(const std::string& filePath, size_t chunkSizeMegaBytes) {
    cuJSONLinesInput input;
    input.data = nullptr;
    input.size = 0;
    size_t chunkSizeBytes = chunkSizeMegaBytes * 1024 * 1024; // Convert MB to bytes

    // ----------------- LOAD FILE -----------------
    std::ifstream file(filePath, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "\033[1;31m Error: Unable to open file: \033[0m \n" << filePath << std::endl;
        return input;
    }

    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // Allocate pinned memory
    uint8_t* h_buffer;
    cudaHostAlloc((void**)&h_buffer, fileSize * sizeof(uint8_t), cudaHostAllocDefault);
    if (!h_buffer) {
        std::cerr << "\033[1;31m Error: Unable to allocate pinned memory! \033[0m \n" << std::endl;
        file.close();
        return input;
    }

    // Read file
    file.read(reinterpret_cast<char*>(h_buffer), fileSize);
    file.close();

    // ----------------- FIND LINE OFFSETS -----------------
    std::vector<size_t> line_offsets;
    line_offsets.push_back(0);  // First line

    for (size_t i = 0; i < fileSize; ++i) {
        if (h_buffer[i] == '\n') {
            line_offsets.push_back(i + 1);  // Start of next line
        }
    }
    if (line_offsets.back() < fileSize) {
        line_offsets.push_back(fileSize);
    }

    input.data = h_buffer;
    input.size = fileSize;

    size_t current_chunk_start = 0;
    size_t current_offset = 0;
    size_t chunk_count = 0;
    for (size_t i = 1; i < line_offsets.size(); ++i) {
        size_t line_start = line_offsets[i - 1];
        size_t line_end = line_offsets[i];

        // If adding this line would exceed the chunk size, finalize current chunk
        if ((line_end - current_chunk_start) > chunkSizeBytes) {
            chunk_count++;
            input.chunks.push_back(h_buffer + current_chunk_start);
            input.chunksSize.push_back(current_offset - current_chunk_start);
            current_chunk_start = line_start;
        }

        current_offset = line_end;
    }

    // Add last chunk
    if (current_chunk_start < fileSize) {
        input.chunks.push_back(h_buffer + current_chunk_start);
        input.chunksSize.push_back(fileSize - current_chunk_start);
        chunk_count++;
    }

    input.chunkCount = chunk_count;
    return input;
}
