#include "cujson/cujsonlines.h"

int main(int argc, char **argv) {
    std::string filePath = "./dataset/twitter_sample_small_records.json";
    // Check command-line arguments
    if (argc >= 2) {
        filePath = argv[1];  // XML file path
        cout << "\033[1;36m[INFORM]\033[0m Using custom JSON file from command line: " << filePath << "\n";
    } else {
        std::cout << "\033[1;36m[INFORM]\033[0m Using default JSON file path.\n";
    }



    // Load File
    size_t maxChunkSize = 256 * 1024 * 1024; // 256MB
    cuJSONLinesInput input = loadJSONLines_chunkSizeBytes(filePath, maxChunkSize);
    if (!input.data) {
        std::cout << "\033[1;31m[ERR]\033[0m File loading failed. Please check the file path.\n";
        return EXIT_FAILURE;
    }
    
    cout << "\033[1;32m[INFORM]\033[0m File loaded successfully. Size: " << input.size << " bytes\n";
    cout << "\033[1;32m[INFORM]\033[0m Number of chunks: " << input.chunkCount << "\n";
    cout << "\033[1;32m[INFORM]\033[0m Chunk sizes: ";
    for (size_t i = 0; i < input.chunksSize.size(); ++i) {
        std::cout << input.chunksSize[i] << " ";
    }
    std::cout << "\n";  


    // Parse JSON with cuJSON
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cuJSONResult parsed_tree = parse_json_lines(input);
    if(parsed_tree.structural == nullptr) {
        std::cout << "\033[1;31m[ERR]\033[0m JSON parsing failed.\n";
        cudaFreeHost(input.data);
        return EXIT_FAILURE;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_time_ms;
    cudaEventElapsedTime(&gpu_time_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    std::cout << "\033[1;32m[RESULT]\033[0m GPU parsing time: " << gpu_time_ms << " ms\n";



    // Process the parsed tree as needed
    // For example, you can print the parsed tree size or perform further operations
    std::cout << "\033[1;32m[RESULT]\033[0m Parsed tree size: " << parsed_tree.totalResultSize << " elements\n";


    // print values of parsed_tree.structural
    // std::cout << "\033[1;32m[RESULT]\033[0m Parsed tree structural values:\n";
    // for (int i = 0; i < parsed_tree.totalResultSize && i < 150; ++i) {
    //     std::cout << parsed_tree.structural[i] << " ";
    // }
    // std::cout << "\n";
    
        
    // // Or you can traverse it for the query purpose:
    cuJSONLinesIterator itr = cuJSONLinesIterator(&parsed_tree, filePath.c_str());

    // //TT1
    itr.gotoArrayIndex(0);
    itr.gotoKey("lang");
    std::cout << "\033[1;32m[RESULT]\033[0m Query Value of $[0].lang: " << itr.getValue() << "\n";

    itr.freeJson();


    cudaDeviceReset();
    return 0;
}