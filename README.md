<p align="center">
  <a href="" rel="noopener">
    <img width="200px" height="200px" src="fig/logo-2.png" alt="cuJSON Project Logo">
    <!-- <img width="200px" height="200px" src="fig/logo.jpeg" style="border-radius: 50%; border: 4px solid #2C3E50;" alt="cuJSON Project Logo"> -->
</p>

<h3 align="center">cuJSON: A Highly Parallel JSON Parser for GPUs</h3>

<div align="center">

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![GitHub Issues](https://img.shields.io/github/issues/ashkanvg/cuJSON.svg)](https://github.com/ashkanvg/cuJSON/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/ashkanvg/cuJSON.svg)](https://github.com/ashkanvg/cuJSON/pulls)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

</div>

---

<p align="left"> 
<b>cuJSON is the world's fastest JSON parser, running entirely on your GPU</b>. Forget the old idea that GPUs can't handle complex data parsing—cuJSON proves them wrong. It's built from the ground up to be super parallel, making quick work of everything from validating your data to figuring out its structure.

<b>The result? cuJSON absolutely flies, leaving other top-tier CPU and even existing GPU parsers in the dust</b>. If you're dealing with tons of JSON data, cuJSON is designed to eliminate that bottleneck and speed things up dramatically.
    <br> 
</p>

## 📝 Table of Contents

- [About](#about)
- [Publication](#paper)
- [Datasets](#datasets)
- [Getting Started](#getting_started)
- [Integrating cuJSON into Your Project](#deployment)
- [Query Iterator](#query)

## 🧐 About <a name = "about"></a>

JSON (JavaScript Object Notation) data is widely used in modern computing, yet its parsing performance can be a major bottleneck. Conventional wisdom suggests that GPUs are ill-suited for parsing due to the branch-heavy nature of parsing algorithms. This work challenges that notion by presenting cuJSON, a novel JSON parser built on a redesigned parsing algorithm, specifically tailored for GPU architectures with minimal branching and maximal parallelism.

cuJSON offloads all three key phases of JSON parsing to the GPU: (i) UTF validation, (ii) JSON tokenization, and (iii) nesting structure recognition. Each phase is powered by a highly parallel algorithm optimized for GPUs, effectively leveraging intrinsic GPU functions and high-performance CUDA libraries for acceleration. 
To maximize the parsing speed, the output of cuJSON is also specially designed in a non-conventional way. Finally, cuJSON is able to break key dependencies in the parsing process, making it possible to accelerate the parsing of a single large JSON file effectively. Evaluation shows that cuJSON not only outperforms highly optimized CPU-based parsers like simdjson  and Pison but also surpasses existing GPU-based parsers like cuDF and GPJSON, in terms of both functionality and performance.

## ✍️ Publication - cuJSON <a name="paper"></a>
This repository contains the official source code for the cuJSON paper. All figures and benchmark results presented in the publication can be fully reproduced using the code provided here.

> Ashkan Vedadi Gargary, Soroosh Safari Loaliyan, and Zhijia Zhao. 2025. <a href="https://doi.org/10.1145/3760250.3762222">CuJSON: A Highly Parallel JSON Parser for GPUs</a>. In Proceedings of the 31st ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 1 (ASPLOS '26). Association for Computing Machinery, New York, NY, USA, 85–100. https://doi.org/10.1145/3760250.3762222

For detailed instructions on how to replicate the experimental results and figures from the paper, <b>specially for research purposes and comparison</b>, please refer to the `paper_reproduced/` directory.

Additionally, all the scripts are available at `paper_reproduced/scripts/readme.md`.


## 📂 Datasets <a name = "datasets"></a>
Two sample datasets are included in the `dataset` folder. Large datasets (used in performance evaluation) can be downloaded from https://drive.google.com/drive/folders/1PkDEy0zWOkVREfL7VuINI-m9wJe45P2Q?usp=sharing and placed into the `dataset` folder. Each dataset comes with two formats:

- For JSON Lines, use those datasets that end in `_small_records.json`. 
- For Standard JSON, use those datasets that end in `_large_record.json`.

## 🏁 Getting Started <a name = "getting_started"></a>

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See [deployment](#deployment) for notes on how to deploy the project on a live system.

### Prerequisites
- `g++` (version 7 or newer),
- the CUDA toolkit (release 12.1 or newer; also validated with CUDA 13.3),
- an NVIDIA GPU with compute capability 8.0 for the commands below, and
- a 64-bit Linux system with a command-line shell.

### Building and Running the Examples (Standard JSON)
Follow these steps to compile and run the cuJSON examples:

1. Clone the Repository (if you haven't already):
First, get a copy of the cuJSON project:
```
git clone https://github.com/ashkanvg/cuJSON
cd cuJSON
```

2. Compile `main.cu` (Standard JSON Parsing), <b>check [notes](#notes) for more details</b>:
```bash
nvcc -O3 -std=c++17 -arch=sm_80 main.cu -o cujson_standard.out
```

3. Run (Standard JSON Parsing):
Once cujson_standard.out is compiled, you can execute it. We've included example JSON files in the dataset/ folder for your convenience, <b>check [notes](#notes) for more details about what datasets should you use</b>.
```bash
./cujson_standard.out ./dataset/twitter_sample_large_record.json
```



### Building and Running the Examples (JSON Lines)
Follow these steps to compile and run the cuJSON examples:

1. Clone the Repository (if you haven't already):
```
git clone https://github.com/ashkanvg/cuJSON
cd cuJSON
```

2. Three entry points demonstrate the available JSON Lines chunking APIs.

Compile `main_jsonlines.cu` to split the input into four chunks:
```bash
nvcc -O3 -std=c++17 -arch=sm_80 main_jsonlines.cu -o cujson_jsonlines_count.out
```

Compile `main_jsonlines_chunksize.cu` to limit chunks to 256 MiB expressed in bytes:
```bash
nvcc -O3 -std=c++17 -arch=sm_80 main_jsonlines_chunksize.cu -o cujson_jsonlines_chunksize_bytes.out
```

Compile `main_jsonlines_chunksize_MB.cu` to specify the same limit in megabytes:
```bash
nvcc -O3 -std=c++17 -arch=sm_80 main_jsonlines_chunksize_MB.cu -o cujson_jsonlines_chunksize_mb.out
```

3. Run the binary for the chunking API you want to test:
```bash
./cujson_jsonlines_count.out ./dataset/twitter_sample_small_records.json
./cujson_jsonlines_chunksize_bytes.out ./dataset/twitter_sample_small_records.json
./cujson_jsonlines_chunksize_mb.out ./dataset/twitter_sample_small_records.json
```


### Notes <a name="notes"></a>:
1. You can use any valid standard JSON or JSON Lines file as input. For more extensive testing, refer to the [datasets](#datasets) section for information on using larger 1GB JSON datasets.
2. `-arch=sm_80` targets compute capability 8.0 GPUs such as the NVIDIA A100. Replace `sm_80` with the architecture of your GPU when building for another device. See NVIDIA's [CUDA GPU compute capability list](https://developer.nvidia.com/cuda-gpus).
3. `-std=c++17` selects the C++17 language standard used by the examples.
4. `-O3` enables optimized host and device compilation.
5. The commands intentionally omit `-w` so compiler warnings remain visible during development.



## 🚀 Integrating cuJSON into Your Project <a name = "deployment"></a>

This section guides you on how to incorporate and utilize the cuJSON library within your own C++/CUDA projects for both standard JSON and JSON Lines parsing.


### Core API
To integrate cuJSON, you'll generally follow these steps:

1. Include the cuJSON Source:
Copy the entire `cujson/` directory from this repository into your project's source tree. Ensure your build system (e.g., nvcc compilation) is configured to compile these files and include their headers.

2. Include the Main Header:
In your source files where you intend to use cuJSON, include its primary header:
- For Standard JSON:
```
#include "cujson/cujson.h"
```

- For JSON Lines:
```
#include "cujson/cujsonlines.h"
```


3. Load Your JSON Data:
Before parsing, load the data into a `cuJSONInput` or `cuJSONLinesInput` structure with the API matching the input format and chunking strategy.
- For Standard JSON:
```cpp
std::string filePath = "./dataset/twitter_sample_large_record.json";
cuJSONInput input = loadJSON(filePath);
```
- For JSON Lines, there are multiple cases: 

a. Split based on the number of chunks:
```cpp
size_t chunkCount = 4;
std::string filePath = "./dataset/twitter_sample_small_records.json";
cuJSONLinesInput input = loadJSONLines_chunkCount(filePath, chunkCount);
```
b. Split based on the maximum chunk size in bytes:
```cpp
std::string filePath = "./dataset/twitter_sample_small_records.json";
size_t maxChunkSizeBytes = 256 * 1024 * 1024;
cuJSONLinesInput input = loadJSONLines_chunkSizeBytes(filePath, maxChunkSizeBytes);
```
c. Split based on the maximum chunk size in megabytes:
```cpp
std::string filePath = "./dataset/twitter_sample_small_records.json";
size_t maxChunkSizeMegaBytes = 256;
cuJSONLinesInput input = loadJSONLines_chunkSizeMegaBytes(filePath, maxChunkSizeMegaBytes);
```

4. Parse the JSON Data:
- For Standard JSON:
```
cuJSONResult parsed_array = parse_standard_json(input);
```
- For JSON Lines:
```
cuJSONResult parsed_array = parse_json_lines(input);
```

5. Access Parsed Results (Further Processing)
The `cuJSONResult` structure (described in detail later) contains pointers to parsed data located in host (CPU) memory. You can either implement your own post-processing logic or use the built-in cuJSON query iterator to navigate the parsed result and extract desired values. We elaborate on the [Query Iterator](#query) section for more information.


### Summary of the Load and Parse APIs
| API Method                            | Description |
| :-------------------------------------------- |:------------------------- |
| `cuJSONInput loadJSON(const std::string& filePath)`       |  Loads a Standard JSON file into a `cuJSONInput` structure. |
| `cuJSONLinesInput loadJSONLines_chunkCount(const std::string& filePath, size_t chunkCount)` | Loads a JSON Lines file and splits it into `chunkCount` chunks. |
| `cuJSONLinesInput loadJSONLines_chunkSizeBytes(const std::string& filePath, size_t chunkSizeBytes)` | Loads a JSON Lines file and splits it into chunks based on a maximum chunk size (in bytes). |
| `cuJSONLinesInput loadJSONLines_chunkSizeMegaBytes(const std::string& filePath, size_t chunkSizeMegaBytes)`| Loads a JSON Lines file and splits it into chunks based on a maximum chunk size (in megabytes). |
| `cuJSONResult parse_standard_json(cuJSONInput input)` | Parses a Standard JSON file after it has been loaded into a `cuJSONInput` structure. |
| `cuJSONResult parse_json_lines(cuJSONLinesInput input)` | Parses a JSON Lines file after it has been loaded into a `cuJSONLinesInput` structure. |



### Understanding cuJSONInput, cuJSONLinesInput and cuJSONResult Structures
These are the primary data structures for interacting with the cuJSON parser:
- `cuJSONInput`:
This structure defines the input format expected by the cuJSON parsing functions. It's designed to be a simple wrapper around raw byte data for Standard JSON.
```
struct cuJSONInput {
    uint8_t* data;                          // Pointer to the raw JSON data buffer
    size_t size;                            // The total size (in bytes) of the JSON data in the buffer.
};
```
- `cuJSONLinesInput`:
This structure defines the input format expected by the cuJSON parsing functions. It's designed to be a simple wrapper around raw byte data for JSON Lines.

```
struct cuJSONLinesInput {
    uint8_t* data;                          // pointer to the data buffer
    size_t chunkCount;                      // number of chunks in the parser
    size_t size;                            // size of the input data
    std::vector<uint8_t*> chunks;           // vector of pointers to each chunk
    std::vector<size_t> chunksSize;         // vector of size to each chunk
};
```
- `cuJSONResult`:
This comprehensive structure encapsulates the output of the GPU-based JSON parsing. It provides the necessary pointers and metadata to reconstruct and navigate the parsed JSON hierarchy.

```
struct cuJSONResult {
    uint8_t* inputJSON;                     // Raw JSON pointer metadata
    int chunkCount;                         // Number of parsed chunks
    int bufferSize;                         // Parser/iterator buffer metadata
    std::vector<int> resultSizes;           // Structural result size for each chunk
    std::vector<int> resultSizesPrefix;     // Prefix sums of per-chunk result sizes
    int32_t* structural;                    // Byte positions of JSON structural characters
    int32_t* pair_pos;                      // Matching closing index for each opening structure
    int depth;                              // Maximum nesting-depth metadata
    int totalResultSize;                    // Combined structural output size
    int fileSize;                           // Structural result span, including boundary entries
};
```




## 🗂️ Query Iterator <a name="query"></a>

After parsing, use `cuJSONIterator` for Standard JSON or `cuJSONLinesIterator` for JSON Lines. Both iterators use the precomputed `pair_pos` and `structural` arrays to skip nested structures and navigate to keys or array elements efficiently.


### Core APIs
The iterator classes share the core navigation methods, with two additional helpers available only for JSON Lines.

1. Initialize the Iterator:
Create the iterator matching the parser result and pass the original file path as a C string.

```cpp
cuJSONIterator standardIterator(&parsed_array, filePath.c_str());
cuJSONLinesIterator linesIterator(&parsed_array, filePath.c_str());
```

2. Traverse and Extract Data: Once initialized, you can use the iterator's methods to move through the JSON structure and retrieve information.

| API Method | Availability | Description |
| :--- | :--- | :--- |
| `int gotoKey(std::string key)` | Both | Moves to the value associated with `key` in the current object. |
| `int gotoArrayIndex(int index)` | Both | Moves to an element in the current array. |
| `int increamentIndex(int index)` | Both | Advances by `index` positions in the structural array. This spelling is part of the current public API. |
| `int gotoNextSibling(int index)` | JSON Lines | Moves to another element in the current array. |
| `bool checkKeyValue(std::string key, std::string value)` | JSON Lines | Tests whether the current object contains the key-value pair. |
| `std::string getKey()` | Both | Returns the key at the current iterator position. |
| `std::string getValue()` | Both | Returns the value at the current iterator position. |
| `void reset()` | Both | Resets the iterator to the first structural entry. |
| `void freeJson()` | Both | Releases the iterator and parsed-result allocations. |

For the bundled Standard JSON sample, enter the synthetic root and then the first array element before selecting `lang`:
```cpp
standardIterator.gotoArrayIndex(0);
standardIterator.gotoArrayIndex(0);
standardIterator.gotoKey("lang");
std::string value = standardIterator.getValue();
```

For the bundled JSON Lines sample, select the first record and then its top-level `lang` field:
```cpp
linesIterator.gotoArrayIndex(0);
linesIterator.gotoKey("lang");
std::string value = linesIterator.getValue();
```

3. Call `freeJson()` on the iterator when processing is complete:
```cpp
standardIterator.freeJson();
// or
linesIterator.freeJson();
```

### Examples
For a variety of usage examples, please refer to the files located in the `paper_reproduced/query_example/` directory.







<!-- 
If you use cuJSON in your research or project, please cite the corresponding paper. A BibTeX entry is provided below for your convenience.
```
@article{YourPaperReference,
  author    = {Ashkan Vedadi Gargary, Soroosh Safari Loaliyan, and Zhijia Zhao},
  title     = {cuJSON: A Highly Parallel JSON Parser for GPUs},
  journal   = {[Your Conference/Journal Name]},
  year      = {[Year of Publication]},
  volume    = {[Volume (if applicable)]},
  number    = {[Number (if applicable)]},
  pages     = {[Page Range]},
  doi       = {[DOI (if available)]},
  url       = {https://github.com/ashkanvg/cuJSON}
}
``` 
-->


<!-- ## ✍️ Authors <a name = "authors"></a>

- [@kylelobo](https://github.com/kylelobo) - Idea & Initial work

See also the list of [contributors](https://github.com/kylelobo/The-Documentation-Compendium/contributors) who participated in this project. -->
<!-- 
## 🎉 Cite <a name = "acknowledgement"></a>

- Hat tip to anyone whose code was used
- Inspiration
- References -->
