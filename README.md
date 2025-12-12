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
- `g++` (version 7 or better), 
- `Cuda` compilation tools (release 12.1), 
- and a 64-bit system with a command-line shell (e.g., Linux, macOS, FreeBSD). 

### Building and Running the Examples (Standard JSON)
Follow these steps to compile and run the cuJSON examples:

1. Clone the Repository (if you haven't already):
First, get a copy of the cuJSON project:
```
git clone https://github.com/ashkanvg/cuJSON
cd cuJSON
```

2. Compile `main.cu` (Standard JSON Parsing), <b>check [notes](#notes) for more details</b>:
```
nvcc -O3 main.cu -o cujson_standard.out -w -gencode=arch=compute_61,code=sm_61
```

3. Run (Standard JSON Parsing):
Once cujson_standard.out is compiled, you can execute it. We've included example JSON files in the dataset/ folder for your convenience, <b>check [notes](#notes) for more details about what datasets should you use</b>.
```
./cujson_standard.out ./dataset/twitter_sample_large_record.json
```



### Building and Running the Examples (JSON Lines)
Follow these steps to compile and run the cuJSON examples:

1. Clone the Repository (if you haven't already):
```
git clone https://github.com/ashkanvg/cuJSON
cd cuJSON
```

2. We have provided multiple files for differenet APIs. We will explore more about the difference of them in the next section. 

Compile `main_jsonlines.cu` for Parsing JSON Lines by splitting the data into 4 chunks, <b>check [notes](#notes) for more details</b>:
```
nvcc -O3 main_jsonlines.cu -o cujson_jsonlines.out -w -gencode=arch=compute_61,code=sm_61
```

Or compile `main_jsonlines_chunksize_MB.cu` or `main_jsonlines_chunksize.cu` for Parsing JSON Lines by splitting the data into chunks with size of `256MB`:
```
nvcc -O3 main_jsonlines_chunksize_MB.cu -o cujson_jsonlines.out -w -gencode=arch=compute_61,code=sm_61
```
Or
```
nvcc -O3 main_jsonlines_chunksize.cu -o cujson_jsonlines.out -w -gencode=arch=compute_61,code=sm_61
```

3. Run (JSON Lines Parsing):
Once `cujson_jsonlines.out` is compiled, you can execute it. We've included example JSON files in the `dataset/` folder for your convenience.
```
./cujson_jsonlines.out ./dataset/twitter_sample_small_records.json
```


### Notes <a name="notes"></a>:
1. You can use any valid standard JSON or JSON Lines file as input. For more extensive testing, refer to the [datasets](#datasets) section for information on using larger 1GB JSON datasets.
2. [-gencode=arch=compute_61,code=sm_61]: (Optional) This flag is for specifying a target GPU architecture. You should replace 61 with the compute capability of your target GPU to achieve optimal performance. For example, for a Turing GPU (RTX 20 series), it might be `compute_75,code=sm_75`. If omitted, nvcc will try to detect your GPU or compile for a generic architecture, which might result in less optimal performance. [You can typically find your GPU's compute capability online here](https://developer.nvidia.com/cuda-gpus).
3. -w: Suppresses all warnings (useful for cleaner output, but be cautious in development).
4. -std=c++17: You can run the compilation using the C++17 standard, which is often required for modern CUDA code.
5. Passes the -O3 optimization flag to the host C++ compiler, ensuring highly optimized CPU code.



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
Before parsing, your JSON data needs to be loaded into a `cuJSONInput` or `cuJSONLinesInput` structure. This structure is a simple container for your raw JSON byte buffer and its size. The `loadJSON` or `loadJSONLines` helper function (presumably provided within the cujson source) is designed for this.
- For Standard JSON:
```
std::string filePath = "./dataset/twitter_sample_large_record.json";
cuJSONInput input = loadJSON(filePath);
```
- For JSON Lines, there are multiple cases: 

a. Split based on the number of chunks:
```
int chunk_counts = 4;
std::string filePath = "./dataset/twitter_sample_small_records.json";
cuJSONLinesInput input = loadJSONLines(filePath, chunk_counts);
```
b. Split based on the maximum chunk size in bytes:
```
string filePath = "./dataset/twitter_sample_small_records.json";
int maxChunkSize = 256 * 1024 * 1024; // 256MB
cuJSONLinesInput input = loadJSONLines_chunkSizeBytes(filePath, maxChunkSize);
```
c. Split based on the maximum chunk size in megabytes:
```
string filePath = "./dataset/twitter_sample_small_records.json";
int maxChunkSize = 256; // 256MB
cuJSONLinesInput input = loadJSONLines_chunkSizeMegaBytes(filePath, maxChunkSize);
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
| `cuJSONInput loadJSON(const string& filePath)`       |  Loads a Standard JSON file into a `cuJSONInput` structure. |
| `cuJSONLinesInput loadJSONLines_chunkCount(const string& filePath, size_t chunkCount)` | Loads a JSON Lines file and splits it into `chunkCount` chunks. |
| `cuJSONLinesInput loadJSONLines_chunkSizeBytes(const string& filePath, size_t chunkSizeBytes)` | Loads a JSON Lines file and splits it into chunks based on a maximum chunk size (in bytes). |
| `cuJSONLinesInput loadJSONLines_chunkSizeMegaBytes(const string& filePath, size_t chunkSizeMegaBytes)`| Loads a JSON Lines file and splits it into chunks based on a maximum chunk size (in megabytes). |
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
    uint8_t* inputJSON;                     // A pointer to the original JSON data buffer.
    int32_t* structural;                    // It containes the byte positions of all JSON structural characters 
    int32_t* pair_pos;                      // It stores the closing structural character's index (in `structural`)
                                            // for the opening structural character at `structural[i]`. 
                                            // This enables efficient navigation of nesting.
    int depth;                              // The maximum nesting depth encountered in the parsed JSON file.
    int totalResultSize;                    // The total size of the combined parsed output.
    int fileSize;                           // The size of the original input JSON file in bytes.
};
```




## 🗂️ Query Iterator <a name="query"></a>

After cuJSON has efficiently parsed your JSON data into the `cuJSONResult` structure on the GPU, the next step is to actually access and extract the values you need. The `cuJSONIterator` is designed precisely for this, allowing you to traverse the parsed JSON array with exceptional speed. Thanks to the pre-calculated `pair_pos` and `structural` array within `cuJSONResult`, the iterator can cleverly skip over entire nested child structures, jumping directly to siblings or specific keys/indices, significantly accelerating data retrieval.


### Core APIs
The cuJSONIterator provides a set of intuitive APIs for navigating the parsed JSON array.

1. Initialize the Iterator:
First, create an instance of the cuJSONIterator, linking it to your `parsed_array` (the output from `parse_standard_json` or `parse_json_lines`) and the original file path.


```
// Assuming 'parsed_array' is your cuJSONResult and 'filePath' is a const char* to the original file path
cuJSONIterator itr = cuJSONIterator(&parsed_array, filePath);
```

2. Traverse and Extract Data: Once initialized, you can use the iterator's methods to move through the JSON structure and retrieve information.


| API Method                            | Description |
| :-------------------------------------------- |:------------------------- |
| `int gotoKey(string key)`       | Attempts to move the iterator's position to the value associated with the specified `key` within the current JSON object. Returns `structural` index on success.
| `int gotoArrayIndex(int index)` | Attempts to move the iterator's position to the element at the specified `index` within the current JSON array. Returns `structural` index on success.|
| `int incrementIndex(int index)` | Advances the iterator's current position forward by `index` steps along the `structural` array. This is useful for sequential traversal within structural array. Returns `structural` index on success. |
| `int gotoNextSibling(int index)` | Advances the iterator's current position forward by `index` steps along the `structural` array. This is useful for sequential traversal within **only** JSON arrays. Returns `structural` index on success. |
| `bool checkKeyValue(string key, string value)` | check if the current object has the key-value pair. Returns `true` on success and `false` on failure. |
| `string getKey()`               | When positioned at a key-value pair within a JSON object, this returns the `string` representation of the current key.|
| `string getValue()`             | When positioned at a value, this returns its `string` representation. |
| `void reset()`                  | Resets the iterator's internal pointer back to the very first structural character of the parsed JSON array, allowing you to re-traverse from the beginning.|


The primary distinction when querying JSON Lines compared to standard JSON lies in the initial step: for Standard JSON, you are required to call gotoArrayIndex(0) at the beginning of your traversal. After this initial call, you can proceed with the same querying steps as you would for a JSON Lines document.


3. Third, make sure to call `freeJSON()` at the end, to free the allocated memory of the input and parsed array.
```
freeJson();                            
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
