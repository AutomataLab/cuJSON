#include "parse_standard_json.h"         // Include the standard parse header
#include "cujson_types.h"
#include <string.h>

// prev1            --> 4 character
// result           --> source
// size             --> total size of array
// total_padded_32  --> based on size howmany thread work on that
__device__ __forceinline__
void vectorizedClassification(uint32_t block_compressed, uint32_t prev1, uint32_t& result, uint64_t size, int total_padded_32){
    constexpr const uint8_t TOO_SHORT   = 1<<0; // 00000001
                                                // The leading byte must be followed by N-1 continuation bytes, 
                                                // where N is the UTF-8 character length.
                                                // 11______ 0_______
                                                // 11______ 11______

    constexpr const uint8_t TOO_LONG    = 1<<1; // The leading byte must not be a continuation byte.
                                                // 0_______ 10______

    constexpr const uint8_t OVERLONG_2  = 1<<5; // Above U+7F for two-byte characters,
                                                // 1100000_ 10______
    constexpr const uint8_t OVERLONG_3  = 1<<2; // Above U+7FF for three-byte characters,
                                                // 11100000 100_____
    constexpr const uint8_t OVERLONG_4  = 1<<6; // Above U+7FFF for three-byte characters,
                                                // 11110000 1000____

    constexpr const uint8_t SURROGATE   = 1<<4; // The decoded character must be not be in U+D800...DFFF
                                                // 11101101 101_____

    constexpr const uint8_t TWO_CONTS   = 1<<7; // Two continious bit after each other
                                                // 10______ 10______

    constexpr const uint8_t TOO_LARGE   = 1<<3; // The decoded character must be less than or equal to U+10FFFF
                                                // 11110100 1001____
                                                // 11110100 101_____
                                                // 11110101 1001____
                                                // 11110101 101_____
                                                // 1111011_ 1001____
                                                // 1111011_ 101_____
                                                // 11111___ 1001____
                                                // 11111___ 101_____

    constexpr const uint8_t TOO_LARGE_1000 = 1<<6;
                                                // Out of the range, it must be maximum 100 if you see 0101, 011_, or 1___
                                                // 11110101 1000____
                                                // 1111011_ 1000____
                                                // 11111___ 1000____


    constexpr const uint8_t CARRY = TOO_SHORT | TOO_LONG | TWO_CONTS; 
                                                // These all have ____ in byte 1 . 10000011


    
    // SIMDJSON use table in CPU, but in GPU Table is very slow
    // we check 4 character in a single time by this:
    constexpr const uint32_t TOO_SHORT_32 = (
        ((uint32_t)TOO_SHORT)       | 
        ((uint32_t)TOO_SHORT) << 8  | 
        ((uint32_t)TOO_SHORT) << 16 | 
        ((uint32_t)TOO_SHORT) << 24
    );
    constexpr const uint32_t TOO_LONG_32 = (
        ((uint32_t)TOO_LONG)        | 
        ((uint32_t)TOO_LONG) << 8   |
        ((uint32_t)TOO_LONG) << 16  |
        ((uint32_t)TOO_LONG) << 24
    );
    constexpr const uint32_t OVERLONG_2_32 = (
        ((uint32_t)OVERLONG_2)       | 
        ((uint32_t)OVERLONG_2) << 8  | 
        ((uint32_t)OVERLONG_2) << 16 | 
        ((uint32_t)OVERLONG_2) << 24
    );
    constexpr const uint32_t OVERLONG_3_32 = (
        ((uint32_t)OVERLONG_3)       | 
        ((uint32_t)OVERLONG_3) << 8  | 
        ((uint32_t)OVERLONG_3) << 16 | 
        ((uint32_t)OVERLONG_3) << 24
    );
    constexpr const uint32_t OVERLONG_4_32 = (
        ((uint32_t)OVERLONG_4)       | 
        ((uint32_t)OVERLONG_4) << 8  | 
        ((uint32_t)OVERLONG_4) << 16 | 
        ((uint32_t)OVERLONG_4) << 24
    );
    constexpr const uint32_t SURROGATE_32 = (
        ((uint32_t)SURROGATE)       | 
        ((uint32_t)SURROGATE) << 8  | 
        ((uint32_t)SURROGATE) << 16 | 
        ((uint32_t)SURROGATE) << 24
    );
    constexpr const uint32_t TWO_CONTS_32 = (
        ((uint32_t)TWO_CONTS)       | 
        ((uint32_t)TWO_CONTS) << 8  | 
        ((uint32_t)TWO_CONTS) << 16 | 
        ((uint32_t)TWO_CONTS) << 24
    );
    constexpr const uint32_t TOO_LARGE_32 = (
        ((uint32_t)TOO_LARGE)       | 
        ((uint32_t)TOO_LARGE) << 8  | 
        ((uint32_t)TOO_LARGE) << 16 | 
        ((uint32_t)TOO_LARGE) << 24
    );
    constexpr const uint32_t TOO_LARGE_1000_32 = (
        ((uint32_t)TOO_LARGE_1000)       | 
        ((uint32_t)TOO_LARGE_1000) << 8  | 
        ((uint32_t)TOO_LARGE_1000) << 16 | 
        ((uint32_t)TOO_LARGE_1000) << 24
    );
    constexpr const uint32_t CARRY_32 = (
        ((uint32_t)CARRY)       | 
        ((uint32_t)CARRY) << 8  | 
        ((uint32_t)CARRY) << 16 |
        ((uint32_t)CARRY) << 24
    );
    



    // [2_high,  ] [1_high ,1_low] <--
    // [a,b,c,d,e,f,g,h] --shr-->   [0,0,0,0,a,b,c,d]
    //                              [0,0,0,0,1,0,0,0] {08}
    //--> kochak tr bashe -->       [1,1,1,1,1,1,1,1]
    //                              [0,0,0,0,0,0,1,0] {too long}

    uint32_t prev1_current = prev1;
    uint32_t byte_1 = 
        (__vcmpltu4(prev1_current, 0x80808080) & TOO_LONG_32) |
        (__vcmpgeu4(prev1_current, 0xC0C0C0C0) & TOO_SHORT_32) | 
        ( (__vcmpeq4(prev1_current, 0xC0C0C0C0) | __vcmpeq4(prev1_current, 0xC1C1C1C1)) & OVERLONG_2_32) | 
        (__vcmpeq4(prev1_current, 0xEDEDEDED) & (SURROGATE_32)) | 
        (__vcmpeq4(prev1_current, 0xE0E0E0E0) & (OVERLONG_3_32)) | 
        (__vcmpeq4(prev1_current, 0xF0F0F0F0) & (OVERLONG_4_32)) | 
        (__vcmpgtu4(prev1_current, 0xF4F4F4F4) & TOO_LARGE_1000_32) | 
        (__vcmpgtu4(prev1_current, 0xF3F3F3F3) & TOO_LARGE_32);
    byte_1 = 
        (__vcmpeq4(byte_1, 0x00000000) & TWO_CONTS_32);
        // (__vcmpgeu4(prev1_current, 0x80808080) & __vcmpltu4(prev1_current, 0xC0C0C0C0) & TWO_CONTS_32);
        // (__vcmpeq4(0x80808080 & prev1_current, 0x80808080) & TWO_CONTS_32);



    uint32_t block_compressed_high = (block_compressed >> 4) & 0x0F0F0F0F; 
    // 4 khune bala ro brdshti 
    // baraye moqaysee adadi bordim daste rast k rahat tr bashe


    // to make it more easier than before, save it and use it multiple time
    uint32_t less_than_12 = __vcmpltu4(block_compressed_high, 0x0C0C0C0C);
    uint32_t byte_2_high = 
        ((__vcmpltu4(block_compressed_high, 0x08080808) | __vcmpgtu4(block_compressed_high, 0x0B0B0B0B)) & TOO_SHORT_32) |
        (less_than_12 & __vcmpgeu4(block_compressed_high, 0x08080808) & (TOO_LONG_32 | OVERLONG_2_32 | TWO_CONTS_32)) | 
        (less_than_12 & __vcmpgtu4(block_compressed_high, 0x08080808) & TOO_LARGE_32) | 
        (__vcmpeq4(block_compressed_high, 0x08080808) & (TOO_LARGE_1000_32 | OVERLONG_4_32)) | 
        (__vcmpgtu4(block_compressed_high, 0x09090909) & less_than_12 & SURROGATE_32); 


    // result =   (byte_1_high & byte_1_low & byte_2_high); 
    result =   (byte_1 & byte_2_high); 
    
    // 0 --> okay and return secussfuly
}

// make sure it has 2 or 3 continuation
// for 3,4 Byte
__device__ __forceinline__
void continuationBytes(uint32_t prev2, uint32_t prev3, uint32_t sc, uint32_t& must32Upper_sc, uint64_t size, int total_padded_32){
    static const uint32_t third_subtract_byte =  
    // 11100000 - 1 --> 11011111 --> This is the maximum of 2 Byte, So if it’s more than this, we have 3 
        (0b11100000u-1)       | 
        (0b11100000u-1) << 8  | 
        (0b11100000u-1) << 16 | 
        (0b11100000u-1) << 24;

    static const uint32_t fourth_subtract_byte = 
        (0b11110000u-1)       | 
        (0b11110000u-1) << 8  |
        (0b11110000u-1) << 16 | 
        (0b11110000u-1) << 24;


    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // the latest byte in our UTF8Bytes (character) is third or fourth
    // subtract prev2 and prev3 from third_subtract_byte and fourth_subtract_byte
    // must be 0 
    // unsign saturated subtraction 4 Byte --> 4 Byte ro parallel az ham kam mikone ya 0 mishe ya 1 
    // ma mikhaym prev2 az third_subtract_byte va prev3 az forth_subtract_byte kochak tr bashe k javab 0 bashe
    uint32_t is_third_byte  = __vsubus4(prev2, third_subtract_byte);
    uint32_t is_fourth_byte = __vsubus4(prev3, fourth_subtract_byte);


    uint32_t gt = ( __vsubss4((int32_t)(is_third_byte | is_fourth_byte), int32_t(0)) ) & 0xFFFFFFFF; 
    
    // because we are working in 32 bit, we need do this for all 4 characters
    uint32_t must32 = __vcmpgtu4(gt, 0); // gt --> hamin must32 hast o mitonim hazfesh knim

    must32Upper_sc = (must32 & 0x80808080) ^ sc;            //  sc --> output of 32 bit check
    // upper bit of each 4 character
} 

__global__ 
void checkAscii(uint32_t* blockCompressed_GPU, uint64_t size, int total_padded_32, bool* hastUTF8, int WORDS){
    int threadId = threadIdx.x;
    __shared__ uint32_t shared_flag;
    
    if(threadId == 0) shared_flag = 0;
    __syncthreads();

    int index = blockIdx.x * blockDim.x + threadId;
    int stride = blockDim.x * gridDim.x;

    for(long i = index; i< total_padded_32; i+=stride){
        int start = i*WORDS;
        #pragma unroll
        for(int j=start; j<size && j<start+WORDS; j++){
            if((blockCompressed_GPU[j] & 0x80808080) != 0) atomicOr(&shared_flag, 1); 
            // check the upper bit
            // atomic or because it works in parallel
        }
        __syncthreads();
    }
    if(threadId == 0 && shared_flag) *hastUTF8 = true;
}

__global__
void checkUTF8(uint32_t* blockCompressed_GPU, uint32_t* error_GPU, uint64_t size, int total_padded_32, int WORDS){
    /*
    - blockCompressed_GPU is a pointer to the compressed data block in GPU memory, 
    - error_GPU is a pointer to a location in GPU memory where the function will store an error code if it detects invalid UTF-8, 
    - size is the size of the data block, 
    - total_padded_32 is the total number of 32-bit words in the padded data block, and 
    - WORDS is the number of words processed by each thread in each iteration of the loop
    */
    static const uint32_t max_val = 
        (uint32_t)(0b11000000u-1 << 24) | 
        (uint32_t)(0b11100000u-1 << 16) | 
        (uint32_t)(0b11110000u-1 << 8)  | 
        (uint32_t)(255); 

    int threadId = threadIdx.x;
    __shared__ uint32_t shared_error;
    if(threadId == 0) shared_error = 0;

    __syncthreads();
    int index = blockIdx.x * blockDim.x + threadId;
    int stride = blockDim.x * gridDim.x;

    for(long i = index; i< total_padded_32; i+=stride){
        int start = i*WORDS;
        #pragma unroll
        for(int j=start; j<size && j<start+WORDS; j++){
            uint32_t current = blockCompressed_GPU[j];
            uint32_t previous = j>0 ? blockCompressed_GPU[j-1] : 0;
            uint32_t prev_incomplete = __vsubus4(previous, max_val);
            
            if((current & 0x80808080) == 0) {
                atomicExch(&shared_error, prev_incomplete);
            }else{
                uint32_t prev1, prev2, prev3;
                uint32_t sc;
                uint32_t must32Upper_sc;

                uint64_t dist = ( ((uint64_t)current) << 32) | (uint64_t) previous;
                prev1 = (uint32_t)(dist >> 3*8); // shifted by 3 byte (3 * 8 bits)
                prev2 = (uint32_t)(dist >> 2*8); // shifted by 2 byte (2 * 8 bits)
                prev3 = (uint32_t)(dist >> 1*8); // shifted by 1 byte (1 * 8 bits)

                vectorizedClassification(current, prev1, sc, size, total_padded_32); // check 1,2 Byte 
                continuationBytes(prev2, prev3, sc, must32Upper_sc, size, total_padded_32); // Check 3,4 byte

                atomicExch(&shared_error, must32Upper_sc); // return error
            }
        }
    }
    __syncthreads();
    if(threadId==0 && shared_error) *error_GPU = shared_error;
}

inline bool UTF8Validation(uint32_t * block_GPU, uint64_t size){
    // _________________INIT_________________________
    int total_padded_32 = size;

    uint32_t* general_ptr;
    cudaMallocAsync(&general_ptr, sizeof(uint32_t), 0);
    uint32_t* error_GPU = general_ptr;
    cudaMemsetAsync(error_GPU, 0, sizeof(uint32_t), 0);

  
    int total_padded_16B = (size+3)/4;
    int WORDS = 4;
    int numBlock_16B = (total_padded_16B+BLOCKSIZE-1) / BLOCKSIZE;


    bool hastUTF8 = false;
    bool* hastUTF8_GPU;
    cudaMallocAsync(&hastUTF8_GPU, sizeof(bool), 0);                  //  Allocates Memory on the Device and Returns a Pointer to the Allocated Memory.
    cudaMemsetAsync(hastUTF8_GPU, 0, sizeof(bool), 0);                //  Initializes a Block of Memory on the Device with a Specified Value


    // _________________PART_1_______________________
    checkAscii<<<numBlock_16B, BLOCKSIZE>>>(block_GPU, size, total_padded_16B, hastUTF8_GPU, WORDS);
    cudaStreamSynchronize(0);
    
    cudaMemcpyAsync(&hastUTF8, hastUTF8_GPU, sizeof(bool), cudaMemcpyDeviceToHost, 0);
    //cudaFreeAsync(hastUTF8_GPU, 0);

    if(!hastUTF8){ 
        //printf("There is no utf8 charachter\n");
        cudaFreeAsync(general_ptr, 0);
        //cudaFreeAsync(hastUTF8_GPU, 0);
        return true;
    }


    // _________________PART_2_______________________
    checkUTF8<<<numBlock_16B, BLOCKSIZE>>>(block_GPU, error_GPU, size, total_padded_16B, WORDS);
    cudaStreamSynchronize(0);

    // _________________RESULT_______________________
    // Stop event
    //t cudaEventRecord(stop, 0);
    //t cudaEventSynchronize(stop);
    //t float elapsedTime;
    //t cudaEventElapsedTime(&elapsedTime, start, stop); // that's our time!
    // Clean up:
    //t cudaEventDestroy(start);
    //t cudaEventDestroy(stop);
    //t time_cal.validation_t += elapsedTime;

    uint32_t error = 0;
    cudaMemcpyAsync(&error, error_GPU, sizeof(uint32_t), cudaMemcpyDeviceToHost, 0);
    cudaFreeAsync(general_ptr, 0);
    if(error != 0){ 
        printf("Incomplete ASCII!\n"); 
        //cudaFreeAsync(error_GPU, 0);
        //cudaFreeAsync(hastUTF8_GPU, 0);
        return false;
    }
            
    //udaFreeAsync(error_GPU, 0);
    //cudaFreeAsync(hastUTF8_GPU, 0);
    return true;

}

__global__
void bitMapCreator(uint8_t* block_GPU, uint32_t* outputSlash, uint32_t* outputQuote, uint32_t* op_GPU, uint32_t* newLine_GPU, uint64_t size, int total_padded_32){
    /*
    The purpose of this function is to create bitmaps for different characters in the data block. 
    Each bitmap represents the presence or absence of a specific character at a particular position in the data block.

    For Example:
    Let's say the data block is {"name": "John\/Doe"}. 
        {"name":"John\/Doe"}
    \:  00000000000000100000
    ":  01000010100000000010
    op: 10000001000000000001

    
    Also, let's assume that the data block has a total of 64 characters (so size = 64) and 
    total_padded_32 = 2 (assuming we have 2 32-bit elements after padding).
    The resulting bitmap will be stored in the outputSlash array. 
    The value of output1[0] will represent the bitmap for the first 32 characters, 
    and output1[1] will represent the bitmap for the remaining 32 characters.
    */


    /*
    - block_GPU: A pointer to an array of uint8_t (unsigned 8-bit integers), representing a block of data.
    - outputSlash, outputQuote, op_GPU, and newLine_GPU: Pointers to arrays of uint32_t (unsigned 32-bit integers), which will store the output results.
    - size: A uint64_t (unsigned 64-bit integer) representing the size of the data block.
    - total_padded_32: An int representing the total number of 32-bit elements after padding.
    */


    /*
    The variables blockIdx.x, blockDim.x, and threadIdx.x are used to calculate the current thread's index
    and the stride value (stride) that determines the loop iterations for each thread.
    */
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;


    for(long i = index; i< total_padded_32; i+=stride){
        // loop inside data among blocks
        // The for loop iterates over the range of total_padded_32 with a step size of stride. 
        // This loop distributes the work among different threads to process different parts of the data block.
        int start_position = i*32;
        // the variable start_position is calculated based on the current index (i) to 
        // determine the starting position of the 32-bit segment being processed.

        // These variables will store the bitmaps for different characters:    
        uint32_t res_slash = 0;     //  " / "
        uint32_t res_quote = 0;     //  " " "
        uint32_t res_op = 0;        //  " { } [ ] : ,"
        uint32_t res_newline = 0;   //  " \n "

        for (int j = start_position; j<start_position+32 && j<size; j++){
            // Loop inside blocks between threads
            // Another nested loop iterates over the range from start to start+32 and 
            // ensures that the loop doesn't go beyond the size of the data block (size).
            uint8_t block = block_GPU[j];
            // uint8_t block_low = block & 0x08; // This operation isolates the 4th bit of the block value.

            // |= : bitwise OR operation 
            // << :bitwise SHIF operation
            block == '\\' ? res_slash |= 1 << (j-start_position) : NULL; // or-shif: first or then shift 
            block == '\"' ? res_quote |= 1 << (j-start_position) : NULL;
            block == '\n' ? res_newline |= 1 << (j-start_position): NULL;
            res_op |= ((( //operands
                    block == '{' ||
                    block == '[' ||
                    block == '}' ||
                    block == ']' ||
                    block == ':' ||
                    block == ','
                    ) ? 1 : 0) << (j-start_position)) ;
            
            // res_newline |= ((( // new line
            //         //block == ' ' ||
            //         //block == '\t' ||
            //         //block == '\r'
            //         block == '\n'
            //         ) ? 1 : 0) << (j-start_position)) ;

    
        }

        // creating bit-map for this 4 results-->
        outputSlash[i] = res_slash;      // " \ "
        outputQuote[i] = res_quote;      // " " "
        op_GPU[i] = res_op;           // operands
        newLine_GPU[i] = res_newline;   // \n
    }

    /*
    Example of how it works:
    Suppose we have the following data block: 
        "Hello\\World".
    We want to create a bitmap to represent the backslash character '\\' in the data block.
    Let's assume that we are currently processing the 10th character in the data block, and (j-start_position) is 10.

    Now, when we reach the 10th character, which is the backslash '\\', the condition block == '\\' will evaluate to true.
    So, the expression res1 |= 1 << (j-start_position) will be executed. 
    Since j-start is 10 in this example, 
        the bitwise left shift operation 1 << (j-start) will shift the number 1 by 10 positions to the left, 
        resulting in the binary number 0b00000000001.

    Then, the bitwise OR operation res_slash |= ... will be performed. 
    Suppose the initial value of res_slash is 0. The operation res1 |= 0b00000000001 will update res1 by setting the 10th bit to 1.
    
    After processing the entire data block, 
    the resulting res_slash bitmap will represent the presence or absence of the backslash character '\\' at each position.

    In the following character if we found any other "\\", 
        the result of this approach again would be the seeting the (j-size_position)th bit to 1.  
    */
}


//     bitMapCreatorSimd<<<numBlock_8, BLOCKSIZE>>>( (uint32_t*) block_GPU, (uint8_t*) backslashes_GPU, (uint8_t*) quote_GPU, (uint8_t*) op_GPU, (uint8_t*) open_close_GPU, size, total_padded_8);

__global__
void bitMapCreatorSimd(uint32_t* block_GPU, uint8_t* outputSlash, uint8_t* outputQuote, uint8_t* op_GPU, uint8_t* open_close_GPU, uint64_t size, int total_padded_8){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = index; i < total_padded_8 && i < size; i += stride) {
        
        int start = i*2;
        
        // if (start < total_padded_8) {
        //     printf("GPU index-%d --> %d\n", start, block_GPU[start]);
        // }
        uint8_t res_slash = 0;     //  " / "
        uint8_t res_quote = 0;     //  " " "
        uint8_t res_op = 0;        //  " { } [ ] : ,"
        uint8_t res_open_close = 0;   //  " \n "

        uint32_t temp_res_slash = 0;
        uint32_t temp_res_quote = 0;
        uint32_t temp_res_op = 0;
        uint32_t temp_colon_comma = 0;
        uint32_t temp_open_close = 0;


        uint32_t block = block_GPU[start];
        // printf("index-%d --> %d\n", i, block_GPU[i]);

        temp_res_slash = (__vcmpeq4(block, 0x5C5C5C5C) & 0x01010101); // 00000000 00000001 00000001 00000001
        temp_res_quote = (__vcmpeq4(block, 0x22222222) & 0x01010101);
        temp_open_close = ((
                    __vcmpeq4(block, 0x5B5B5B5B) |
                    __vcmpeq4(block, 0x5D5D5D5D) |
                    __vcmpeq4(block, 0x7B7B7B7B) |
                    __vcmpeq4(block, 0x7D7D7D7D) ) & 0x01010101);

        // temp_colon_comma_newline = ((
        //             __vcmpeq4(block, 0x3A3A3A3A) |
        //             __vcmpeq4(block, 0x2C2C2C2C) |
        //             __vcmpeq4(block, 0x0A0A0A0A)) & 0x01010101);
        
        temp_colon_comma = ((
                    __vcmpeq4(block, 0x3A3A3A3A) |
                    __vcmpeq4(block, 0x2C2C2C2C)) & 0x01010101);

        temp_res_op = temp_colon_comma | temp_open_close;
    
        
        if(i == total_padded_8 - 1 && 4*(start + 1)  >= size ){
            for(int j = 0; j < 4; j++){
                res_slash   |= (uint8_t) (temp_res_slash >> j*7 & 0x0F) ;
                res_quote   |= (uint8_t) (temp_res_quote >> j*7 & 0x0F);
                res_op      |= (uint8_t) (temp_res_op >> j*7 & 0x0F   );
                res_open_close |= (uint8_t) (temp_open_close >> j*7 & 0x0F );
            }
            outputSlash[i] = res_slash;      // " \ "
            outputQuote[i] = res_quote;      // " " "
            op_GPU[i] = res_op;              // operands
            open_close_GPU[i] = res_open_close;    // \n
            continue;
        }

        uint32_t temp2_res_slash = 0;
        uint32_t temp2_res_quote = 0;
        uint32_t temp2_res_op = 0;
        uint32_t temp2_colon_comma = 0;
        uint32_t temp2_open_close= 0;


      

        uint32_t block_2 = block_GPU[start+1];
        // printf("index-%d -2-> %d\n", i, block_2);
        temp2_res_slash = (__vcmpeq4(block_2, 0x5C5C5C5C) & 0x01010101); // 00000001 00000001 00000001 00000001
        temp2_res_quote = (__vcmpeq4(block_2, 0x22222222) & 0x01010101);

        temp2_open_close = (
                    __vcmpeq4(block_2, 0x5B5B5B5B) |
                    __vcmpeq4(block_2, 0x5D5D5D5D) |
                    __vcmpeq4(block_2, 0x7B7B7B7B) |
                    __vcmpeq4(block_2, 0x7D7D7D7D) ) & 0x01010101;

        // temp2_colon_comma_newline = (
        //             __vcmpeq4(block_2, 0x3A3A3A3A) |
        //             __vcmpeq4(block_2, 0x2C2C2C2C) |
        //             __vcmpeq4(block_2, 0x0A0A0A0A)) & 0x01010101;
        
        temp2_colon_comma = (
                    __vcmpeq4(block_2, 0x3A3A3A3A) |
                    __vcmpeq4(block_2, 0x2C2C2C2C)) & 0x01010101;

        temp2_res_op = temp2_colon_comma | temp2_open_close;


        for(int j = 0; j < 4; j++){
            //   j=0     00000001               | 00000001 << 3 = 00010000
            //   j=1     00000010               | 00000010 << 3
            //   j=2     00000100               | 00000100 << 3
            //   j=3     ...
            res_slash   |= (uint8_t) ((temp_res_slash >> j*7) | ((temp2_res_slash >> j*7) << 4) );
            res_quote   |= (uint8_t) (temp_res_quote >> j*7 | ((temp2_res_quote >> j*7) << 4) );
            res_op      |= (uint8_t) (temp_res_op >> j*7    | ((temp2_res_op >> j*7) << 4) );
            res_open_close |= (uint8_t) (temp_open_close >> j*7 | ((temp2_open_close >> j*7) << 4) );
        }


        // if(index == 15) {
        //     char binary1[33], binary2[33];
        //     u32ToBinary(temp_res_quote, binary1);
        //     u32ToBinary(temp2_res_quote, binary2);
        //     printf("temp1: %s\n", binary1);
        //     printf("temp2: %s\n", binary2);
        //     printf("merge: %s\n", byteToBinary(res_quote));
        // }
        // 0-1: 0
        // 2-3: 1
        // 4-5: 2
        // 6-7: 3
        // cout << "index-" << i << "-->" << res_quote << endl;
        // printf("index-%d --> %d\n", i, res_quote);
        // storing results in the larger arrays
        outputSlash[i] = res_slash;      // " \ "
        outputQuote[i] = res_quote;      // " " "
        op_GPU[i] = res_op;              // operands
        open_close_GPU[i] = res_open_close;    // open/close
    }
}


// must change based on bitMapCreatorSimd
__global__
void bitMapCreatorSimd32(uint32_t* block_GPU, uint32_t* outputSlash, uint32_t* outputQuote, uint32_t* op_GPU, uint32_t* newLine_GPU, uint64_t size, int total_padded_32){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = index; i < total_padded_32 && i < size; i += stride) {
        int start = i*8;
        // if (start < total_padded_32) {
        //     printf("GPU index-%d --> %d\n", start, block_GPU[start]);
        // }
        uint32_t res_slash = 0;     //  " / "
        uint32_t res_quote = 0;     //  " " "
        uint32_t res_op = 0;        //  " { } [ ] : ,"
        uint32_t res_newline = 0;   //  " \n "

        uint32_t temp_res_slash = 0;
        uint32_t temp_res_quote = 0;
        uint32_t temp_res_op = 0;
        uint32_t temp_res_newline = 0;

        uint32_t block = block_GPU[start];
        // printf("index-%d --> %d\n", i, block_GPU[i]);

        temp_res_slash = (__vcmpeq4(block, 0x5C5C5C5C) & 0x01010101); // 00000000 00000001 00000001 00000001
        temp_res_quote = (__vcmpeq4(block, 0x22222222) & 0x01010101);
        temp_res_op = ((__vcmpeq4(block, 0x5B5B5B5B) |
                    __vcmpeq4(block, 0x5D5D5D5D) |
                    __vcmpeq4(block, 0x7B7B7B7B) |
                    __vcmpeq4(block, 0x7D7D7D7D) |
                    __vcmpeq4(block, 0x3A3A3A3A) |
                    __vcmpeq4(block, 0x2C2C2C2C)) & 0x01010101);
        temp_res_newline = (__vcmpeq4(block, 0x32323232) & 0x01010101);

        uint32_t temp2_res_slash = 0;
        uint32_t temp2_res_quote = 0;
        uint32_t temp2_res_op = 0;
        uint32_t temp2_res_newline= 0;

        uint32_t block_2 = block_GPU[start+1];
        // printf("index-%d -2-> %d\n", i, block_2);
        temp2_res_slash = (__vcmpeq4(block_2, 0x5C5C5C5C) & 0x01010101); // 00000001 00000001 00000001 00000001
        temp2_res_quote = (__vcmpeq4(block_2, 0x22222222) & 0x01010101);
        temp2_res_op = ((__vcmpeq4(block_2, 0x5B5B5B5B) |
                    __vcmpeq4(block_2, 0x5D5D5D5D) |
                    __vcmpeq4(block_2, 0x7B7B7B7B) |
                    __vcmpeq4(block_2, 0x7D7D7D7D) |
                    __vcmpeq4(block_2, 0x3A3A3A3A) |
                    __vcmpeq4(block_2, 0x2C2C2C2C)) & 0x01010101);
        temp2_res_newline = (__vcmpeq4(block_2, 0x32323232) & 0x01010101);

        uint32_t temp3_res_slash = 0;
        uint32_t temp3_res_quote = 0;
        uint32_t temp3_res_op = 0;
        uint32_t temp3_res_newline = 0;

        uint32_t block_3 = block_GPU[start+2];
        // printf("index-%d --> %d\n", i, block_GPU[i]);

        temp3_res_slash = (__vcmpeq4(block_3, 0x5C5C5C5C) & 0x01010101); // 00000000 00000001 00000001 00000001
        temp3_res_quote = (__vcmpeq4(block_3, 0x22222222) & 0x01010101);
        temp3_res_op = ((__vcmpeq4(block_3, 0x5B5B5B5B) |
                    __vcmpeq4(block_3, 0x5D5D5D5D) |
                    __vcmpeq4(block_3, 0x7B7B7B7B) |
                    __vcmpeq4(block_3, 0x7D7D7D7D) |
                    __vcmpeq4(block_3, 0x3A3A3A3A) |
                    __vcmpeq4(block_3, 0x2C2C2C2C)) & 0x01010101);
        temp3_res_newline = (__vcmpeq4(block_3, 0x32323232) & 0x01010101);

        uint32_t temp4_res_slash = 0;
        uint32_t temp4_res_quote = 0;
        uint32_t temp4_res_op = 0;
        uint32_t temp4_res_newline = 0;

        uint32_t block_4 = block_GPU[start+3];
        // printf("index-%d --> %d\n", i, block_GPU[i]);

        temp4_res_slash = (__vcmpeq4(block_4, 0x5C5C5C5C) & 0x01010101); // 00000000 00000001 00000001 00000001
        temp4_res_quote = (__vcmpeq4(block_4, 0x22222222) & 0x01010101);
        temp4_res_op = ((__vcmpeq4(block_4, 0x5B5B5B5B) |
                    __vcmpeq4(block_4, 0x5D5D5D5D) |
                    __vcmpeq4(block_4, 0x7B7B7B7B) |
                    __vcmpeq4(block_4, 0x7D7D7D7D) |
                    __vcmpeq4(block_4, 0x3A3A3A3A) |
                    __vcmpeq4(block_4, 0x2C2C2C2C)) & 0x01010101);
        temp4_res_newline = (__vcmpeq4(block_4, 0x32323232) & 0x01010101);

        uint32_t temp5_res_slash = 0;
        uint32_t temp5_res_quote = 0;
        uint32_t temp5_res_op = 0;
        uint32_t temp5_res_newline= 0;

        uint32_t block_5 = block_GPU[start+4];
        // printf("index-%d -2-> %d\n", i, block_2);
        temp5_res_slash = (__vcmpeq4(block_5, 0x5C5C5C5C) & 0x01010101); // 00000001 00000001 00000001 00000001
        temp5_res_quote = (__vcmpeq4(block_5, 0x22222222) & 0x01010101);
        temp5_res_op = ((__vcmpeq4(block_5, 0x5B5B5B5B) |
                    __vcmpeq4(block_5, 0x5D5D5D5D) |
                    __vcmpeq4(block_5, 0x7B7B7B7B) |
                    __vcmpeq4(block_5, 0x7D7D7D7D) |
                    __vcmpeq4(block_5, 0x3A3A3A3A) |
                    __vcmpeq4(block_5, 0x2C2C2C2C)) & 0x01010101);
        temp5_res_newline = (__vcmpeq4(block_5, 0x32323232) & 0x01010101);

        uint32_t temp6_res_slash = 0;
        uint32_t temp6_res_quote = 0;
        uint32_t temp6_res_op = 0;
        uint32_t temp6_res_newline = 0;

        uint32_t block_6 = block_GPU[start+5];
        // printf("index-%d --> %d\n", i, block_GPU[i]);

        temp6_res_slash = (__vcmpeq4(block_6, 0x5C5C5C5C) & 0x01010101); // 00000000 00000001 00000001 00000001
        temp6_res_quote = (__vcmpeq4(block_6, 0x22222222) & 0x01010101);
        temp6_res_op = ((__vcmpeq4(block_6, 0x5B5B5B5B) |
                    __vcmpeq4(block_6, 0x5D5D5D5D) |
                    __vcmpeq4(block_6, 0x7B7B7B7B) |
                    __vcmpeq4(block_6, 0x7D7D7D7D) |
                    __vcmpeq4(block_6, 0x3A3A3A3A) |
                    __vcmpeq4(block_6, 0x2C2C2C2C)) & 0x01010101);
        temp6_res_newline = (__vcmpeq4(block_6, 0x32323232) & 0x01010101);

        uint32_t temp7_res_slash = 0;
        uint32_t temp7_res_quote = 0;
        uint32_t temp7_res_op = 0;
        uint32_t temp7_res_newline= 0;

        uint32_t block_7 = block_GPU[start+6];
        // printf("index-%d -2-> %d\n", i, block_2);
        temp7_res_slash = (__vcmpeq4(block_7, 0x5C5C5C5C) & 0x01010101); // 00000001 00000001 00000001 00000001
        temp7_res_quote = (__vcmpeq4(block_7, 0x22222222) & 0x01010101);
        temp7_res_op = ((__vcmpeq4(block_7, 0x5B5B5B5B) |
                    __vcmpeq4(block_7, 0x5D5D5D5D) |
                    __vcmpeq4(block_7, 0x7B7B7B7B) |
                    __vcmpeq4(block_7, 0x7D7D7D7D) |
                    __vcmpeq4(block_7, 0x3A3A3A3A) |
                    __vcmpeq4(block_7, 0x2C2C2C2C)) & 0x01010101);
        temp7_res_newline = (__vcmpeq4(block_7, 0x32323232) & 0x01010101);


        uint32_t temp8_res_slash = 0;
        uint32_t temp8_res_quote = 0;
        uint32_t temp8_res_op = 0;
        uint32_t temp8_res_newline= 0;

        uint32_t block_8 = block_GPU[start+7];
        // printf("index-%d -2-> %d\n", i, block_2);
        temp8_res_slash = (__vcmpeq4(block_8, 0x5C5C5C5C) & 0x01010101); // 00000001 00000001 00000001 00000001
        temp8_res_quote = (__vcmpeq4(block_8, 0x22222222) & 0x01010101);
        temp8_res_op = ((__vcmpeq4(block_8, 0x5B5B5B5B) |
                    __vcmpeq4(block_8, 0x5D5D5D5D) |
                    __vcmpeq4(block_8, 0x7B7B7B7B) |
                    __vcmpeq4(block_8, 0x7D7D7D7D) |
                    __vcmpeq4(block_8, 0x3A3A3A3A) |
                    __vcmpeq4(block_8, 0x2C2C2C2C)) & 0x01010101);
        temp8_res_newline = (__vcmpeq4(block_8, 0x32323232) & 0x01010101);

        for(int j = 0; j < 4; j++){
            //   j=0     00000001               | 00000001 << 3 = 00010000
            //   j=1     00000010               | 00000010 << 3
            //   j=2     00000100               | 00000100 << 3
            //   j=3     ...
            res_slash   |= ((temp_res_slash >> j*7)          | ((temp2_res_slash >> j*7) << 4) | 
                            ((temp3_res_slash >> j*7) << 8)  | ((temp4_res_slash >> j*7) << 12) |
                            ((temp5_res_slash >> j*7) << 16)  | ((temp6_res_slash >> j*7) << 20) |
                            ((temp7_res_slash >> j*7) << 24) | ((temp8_res_slash >> j*7) << 28) 
                            );
            res_quote   |= ((temp_res_quote >> j*7)          | ((temp2_res_quote >> j*7) << 4) | 
                            ((temp3_res_quote >> j*7) << 8)  | ((temp4_res_quote >> j*7) << 12) |
                            ((temp5_res_quote >> j*7) << 16)  | ((temp6_res_quote >> j*7) << 20) |
                            ((temp7_res_quote >> j*7) << 24) | ((temp8_res_quote >> j*7) << 28) 
                            );
            res_op      |= ((temp_res_op >> j*7)          | ((temp2_res_op >> j*7) << 4) | 
                            ((temp3_res_op >> j*7) << 8)  | ((temp4_res_op >> j*7) << 12) |
                            ((temp5_res_op >> j*7) << 16) | ((temp6_res_op >> j*7) << 20) |
                            ((temp7_res_op >> j*7) << 24) | ((temp8_res_op >> j*7) << 28) 
                            );
            res_newline |= ((temp_res_newline >> j*7)          | ((temp2_res_newline >> j*7) << 4) | 
                            ((temp3_res_newline >> j*7) << 8)  | ((temp4_res_newline >> j*7) << 12) |
                            ((temp5_res_newline >> j*7) << 16) | ((temp6_res_newline >> j*7) << 20) |
                            ((temp7_res_newline >> j*7) << 24) | ((temp8_res_newline >> j*7) << 28) 
                            );
        }


        outputSlash[i] = res_slash;      // " \ "
        outputQuote[i] = res_quote;      // " " "
        op_GPU[i] = res_op;              // operands
        newLine_GPU[i] = res_newline;    // \n
    }
}

__global__
void findEscapedQuoteMerge_NEW(uint32_t* backslashes_GPU, uint32_t* quote_GPU, uint32_t* real_quote_GPU, int size, int total_padded_32, int WORDS){
    /*
        The findEscapedQuote function analyzes the input data block and identifies the escaped characters. 
        It processes the data in parallel, utilizing bitwise operations to detect escape sequences 
        and mark the positions of non-escaped characters. 
        The resulting information is stored in the real_quote_GPU array for further processing or analysis.
    */


    // odd-length sequences of backslashes means we have escape character


    // OVERFLOW IS CAME FROM HIGH OF PREVIOUS WORD TO LOW BIT OF CURRENT WORD
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(long i = index; i< total_padded_32; i+=stride){
        int start = i*WORDS;

        // Parallel-For in GPU: 
        #pragma unroll
        for(int k=start; k<size && k<start+WORDS; k++){
            uint32_t overflow = 2;
            // It is used in combination with bitwise operations to detect --> 2 mean maybe overflow maybe not

            uint32_t evenBits = 0x55555555UL; // 5 --> 0101
            uint32_t oddBits = ~evenBits;
            // This is a uint32_t constant with a value of 0x55555555UL. 
            // It represents a bitmask with 1s in all even bit positions. 

            long j=k-1;
            if(k == 0) overflow = 0;
            uint32_t current_word_quote = quote_GPU[k];
            uint32_t backslashes = backslashes_GPU[k];                          //[0,1,1,1,0,0,0,1]

            uint32_t possible_escaped_quote =  current_word_quote & (backslashes << 1 | 1);  
            // this one is for finding possible escape double qutoes that we have to check
            if(possible_escaped_quote == 0){
                real_quote_GPU[k] = current_word_quote;
                quote_GPU[k] = (uint32_t) __popc(real_quote_GPU[k]);  // quote is total_one, we will rename it
                continue;
            }

            while(overflow == 2){
                uint32_t backslash_j = backslashes_GPU[j];                              //[1,1,1,0,0,0,0,0]
                // This is a uint32_t variable that stores the value of backslashes_GPU[j]. It represents the backslashes at position j in the input data.
                uint8_t following_backslash_counts = __clz(~backslash_j); // Convert to 0-based index
                overflow = (following_backslash_counts == 32) ? 2 : following_backslash_counts & 1; 
                j--; // previous chunk qable 
            }

            // has overflow at this step: 0 or 1
            // as same as SIMDJSON
            backslashes = backslashes & (~overflow);                            //[0,1,1,1,0,0,0,0] 
            uint32_t applyEscapedChar = (backslashes << 1) | overflow;            //[1,1,1,0,0,0,0,1] --> chn amaln yek backslash moaser bode k khonsa mikrde miomde to 

            // All BACKSLASHES that are at ODD LOCATION and not ESCAPED
            uint32_t oddSequence = backslashes & oddBits & ~applyEscapedChar;      
            uint32_t sequenceStartatEven = oddSequence + backslashes;           //[0,1,1,1,0,0,0,0]
            uint32_t invert_mask = sequenceStartatEven << 1;            //[1,1,1,0,0,0,0,0]
            uint32_t escaped = (evenBits ^ invert_mask) & applyEscapedChar;
            
            
            real_quote_GPU[k] = (~escaped) & current_word_quote;    // quote hae vaghie   
            quote_GPU[k] = (uint32_t) __popc(real_quote_GPU[k]);  // quote is total_one, we will rename it
        }
    }
}

// 1 WORD - Step 3:
// 1 Word + popc - scatter
__global__
void reduceChunkBaseline(uint32_t* real_quote_GPU, uint32_t* prediction_GPU, int total_padded_32){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for(long i = index; i < total_padded_32; i+=stride){
        prediction_GPU[i] = (uint32_t) __popc(real_quote_GPU[i]);
    }
}

// CUDA kernel where prefix_xor is called
// inStringFinderBaseline<<<numBlock, BLOCKSIZE>>>(real_quote_GPU, total_one_GPU, inString_GPU, total_padded_32);
__global__
void inStringFinderBaseline(uint32_t* real_quote_GPU, uint32_t* prefix_sum_ones, uint32_t* res, int total_padded_32){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < total_padded_32; i += stride){
        // Check if the prefix sum is odd (overflow)
        bool overflow = prefix_sum_ones[i] & 1;
        // Apply prefix_xor to real_quote_GPU
        res[i] = prefix_xor(real_quote_GPU[i]);
        // Update res[i] based on the overflow
        res[i] = overflow ? ~res[i] : res[i];
    }
}

// 2 WORD - Step 3:
// 2 Word + popc - scatter
__global__
void reduceChunkBaseline64(uint64_t* real_quote_GPU, uint64_t* prediction_GPU, int total_padded_64){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for(long i = index; i < total_padded_64; i+=stride){
        prediction_GPU[i] = (uint64_t) __popcll(real_quote_GPU[i]);
    }
}


// CUDA kernel where prefix_xor is called
__global__
void inStringFinderBaseline64(uint64_t* real_quote_GPU, uint64_t* prefix_sum_ones, uint64_t* res, int total_padded_64){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < total_padded_64; i += stride){
        // Check if the prefix sum is odd (overflow)
        bool overflow = prefix_sum_ones[i] & 1;

        // Apply prefix_xor64 to real_quote_GPU
        res[i] = prefix_xor64(real_quote_GPU[i]);

        // Update res[i] based on the overflow
        res[i] = overflow ? ~res[i] : res[i];
    }
}

__global__
void findOutUsefulString(uint32_t* op_GPU, uint32_t* newLine_GPU, uint32_t* inString_GPU, uint64_t size, int total_padded_32, int WORDS){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // find all useful character such as { } [ ] : , \n out of sring.
    for(long i = index; i< total_padded_32; i+=stride){
        int start = i*WORDS;
        #pragma unroll
        for(int k=start; k<size && k<start+WORDS; k++){
            uint32_t op = op_GPU[k];                    // { } [ ] : ,
            uint32_t newLine = newLine_GPU[k];    // \n
            uint32_t in_string = inString_GPU[k];

            uint32_t usefulCharacter = op | newLine;
            inString_GPU[k] = ~in_string & usefulCharacter;
        }
    }
}

__global__
void findOutUsefulStringMerge(uint32_t* op_GPU, uint32_t* open_close_GPU, uint32_t* inString_GPU, uint64_t size, int total_padded_32, int WORDS, uint32_t* total_bits){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // find all useful character such as { } [ ] : , \n out of sring.
    for(long i = index; i< total_padded_32; i+=stride){
        int start = i*WORDS;
        #pragma unroll
        for(int k=start; k<size && k<start+WORDS; k++){
            uint32_t all_structural = op_GPU[k];                    // { } [ ] : ,
            uint32_t open_close = open_close_GPU[k];    // \n
            uint32_t in_string = inString_GPU[k];

            inString_GPU[k] = ~in_string & all_structural; // all structural that are out string
            open_close_GPU[k] = ~in_string & open_close; // all open close that are out string

            total_bits[k] = (uint32_t) __popc(inString_GPU[k]);  // total_bits is total_one, we will rename it
            op_GPU[k] = (uint32_t) __popc(open_close_GPU[k]);  // total_bits of open_close is total_one, we will rename it, we put it in op_GPU to prevenet new allocation

        }
    }
}

__global__
void count_set_bits(uint32_t* input, uint32_t* total_bits, int size, uint32_t total_padded_32, int WORDS){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(uint32_t i = index; i< total_padded_32; i+=stride){
        //UPdate
        int start = i*WORDS;
        #pragma unroll
        for(int k=start; k<size && k<start+WORDS; k++){
            total_bits[k] = (uint32_t) __popc(input[k]);
        }
    }
}

__global__
void removeCopy( uint32_t* set_bit_count,
                 uint32_t* set_bit_count_open_close,
                 uint32_t* out_string, 
                 uint32_t* open_close_GPU, 
                 uint8_t* block_GPU, 
                //  uint8_t* out_string_8_GPU, 
                 uint32_t* out_string_8_index_GPU, 
                 uint8_t* out_string_open_close_8_GPU, 
                 uint32_t* out_string_open_close_8_index_GPU, 
                 uint32_t size, 
                 uint32_t size_structural, 
                 uint32_t size_open_close, 
                 uint32_t total_padded_32,
                 uint64_t lastStructuralIndex,
                 uint64_t lastChunkIndex){

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // each thread working on 32 bit, and each bit has 32 bits.
    // each bit is a single character
    // we want to convert it from bit to char
    for(uint32_t i = index; i< total_padded_32; i+=stride){
        uint32_t local_out_string = out_string[i];
        if (local_out_string == 0) continue; 

        uint32_t local_out_string_open_close = open_close_GPU[i];


        uint32_t total_before = i > 0 ? set_bit_count[i-1] : 0;
        uint32_t total_before_open_close = i > 0 ? set_bit_count_open_close[i-1] : 0;

        // uint32_t current_total = 0;
        // uint32_t current_total_open_close = 0;
        
        uint32_t current_total = 0;
        uint32_t current_total_open_close = 0;
        
        uint32_t k = i*32; 

        // Calculate first and last set bit positions
        uint8_t first_set_bit_pos = __ffs(local_out_string) - 1; // Convert to 0-based index
        uint8_t last_set_bit_pos = 32 - __clz(local_out_string) - 1; // Convert to 0-based index

        for (int j = first_set_bit_pos; j <= last_set_bit_pos && k + j < size; j++) {
            uint32_t adjusted_index = total_before + current_total;
            uint32_t adjusted_index_open_close = total_before_open_close + current_total_open_close;

            uint8_t current_bit = (local_out_string >> j) & 1;      
            if (current_bit == 1){
                uint8_t currentChar = block_GPU[k + j];
                // uint8_t replacementChar = (currentChar == 0x0A) ? ',' : currentChar; // check if its new line put comma instead, Check for newline character (0x0A) and set a comma in out_string_8_GPU
                // out_string_8_GPU[adjusted_index] = replacementChar; // Set the value in out_string_8_GPU
                out_string_8_index_GPU[adjusted_index] = k + j + 1 + lastChunkIndex;

                uint8_t current_bit_open_close = (local_out_string_open_close >> j) & 1;      
                if(current_bit_open_close == 1){
                    out_string_open_close_8_GPU[adjusted_index_open_close] = currentChar; // Set the value in out_string_8_GPU
                    out_string_open_close_8_index_GPU[adjusted_index_open_close] = adjusted_index;
                }
                current_total_open_close += current_bit_open_close;       // prefix_sum inside count_set_bits_open_close

            }
            current_total += current_bit;       // prefix_sum inside count_set_bits

        }  
    }
}

inline uint8_t * Tokenize(  uint8_t* block_GPU, 
                            uint64_t size, 
                            int &ret_size, 
                            uint32_t  &last_index_tokens, 
                            uint32_t  &last_index_tokens_open_close, 
                            uint32_t* &in_string_out_index_d,
                            // uint8_t*  &open_close_d,
                            uint32_t* &open_close_index_d,
                            uint64_t lastStructuralIndex,               // last structural index from previous chunk
                            uint64_t lastChunkIndex                     // last real json index from previous chunk
                            ){
    int total_padded_32 = (size+31)/32 ; // size be byte eshe totall padded be bit eshe
    uint8_t*  open_close_d;
    // +31 vase ine k 0 be ma nade o min 1 bde bema
    // va msln size=6 bashe --> 
    //int numBlockBySize = (size + BLOCKSIZE - 1) / BLOCKSIZE;

    // ____________________Initialize________________________
    // uint32_t* general_ptr;
    // cudaMallocAsync(&general_ptr, total_padded_32*sizeof(uint32_t)*ROW5, 0);
    // uint32_t* quote_GPU = general_ptr;
    // uint32_t* backslashes_GPU = general_ptr+total_padded_32;
    // uint32_t* newLine_GPU = general_ptr+total_padded_32*ROW2;
    // uint32_t* op_GPU = general_ptr+total_padded_32*ROW3;

    // int WORDS = 2;
    // int total_padded_32B = (size+7)/8;
    // int total_padded_8B = (total_padded_32+1)/2;
    // int total_padded_32_div_32 = (total_padded_32+31)/32;
    // int smallNumBlock = (total_padded_32_div_32 + BLOCKSIZE - 1) / BLOCKSIZE;
    // int numBlock_8B = (total_padded_8B+BLOCKSIZE-1) / BLOCKSIZE;
    // int numBlock_32B    = (total_padded_32B+BLOCKSIZE-1) / BLOCKSIZE;

    // 32 * 8 --> 32

    uint32_t* general_ptr;
    cudaMallocAsync(&general_ptr, total_padded_32 * sizeof(uint32_t) * ROW5, 0);
    for (int i = 0; i < ROW5; ++i) {
        uint32_t* row_end_ptr = general_ptr + i * total_padded_32 + (total_padded_32 - 1);
        cudaMemsetAsync(row_end_ptr, 0, sizeof(uint32_t), 0);
    }
    cudaStreamSynchronize(0);

    uint32_t* quote_GPU         = general_ptr;
    uint32_t* backslashes_GPU   = general_ptr + total_padded_32;
    uint32_t* open_close_GPU    = general_ptr + total_padded_32 * ROW2;
    uint32_t* op_GPU            = general_ptr + total_padded_32 * ROW3;

    int WORDS = 2;

    int total_padded_8B         = (total_padded_32 + 1) / 2;
    int total_padded_16B        = (total_padded_32 + 3) / 4;
    int total_padded_32_div_8   = (total_padded_32 + 7) / 8;
    int total_padded_32_div_32  = (total_padded_32 + 31) / 32;

    int total_padded_8 = (size + 7) / 8;
    int total_padded_32B = (size + 7) / 8;
    // int total_padded_32 = (size + 31) / 32; // most used
    int total_padded_64 = (size + 63) / 64;

    int smallNumBlock   = (total_padded_32_div_32 + BLOCKSIZE - 1) / BLOCKSIZE;
    int smallNumBlock_8 = (total_padded_32_div_8 + BLOCKSIZE - 1) / BLOCKSIZE;

    int numBlock        = (total_padded_32 + BLOCKSIZE - 1) / BLOCKSIZE;
    int numBlock_8      = (total_padded_8 + BLOCKSIZE - 1) / BLOCKSIZE;
    int numBlock_8B     = (total_padded_8B + BLOCKSIZE - 1) / BLOCKSIZE;
    int numBlock_16B    = (total_padded_16B + BLOCKSIZE - 1) / BLOCKSIZE;
    int numBlock_32B    = (total_padded_32B + BLOCKSIZE - 1) / BLOCKSIZE;
    int numBlock_64     = (total_padded_64 + BLOCKSIZE - 1) / BLOCKSIZE;


    // Prepare
    //t cudaEvent_t start, stop;
    //t cudaEventCreate(&start);
    //t cudaEventCreate(&stop);
    // Start record
    //t cudaEventRecord(start, 0);

    // auto start = chrono::high_resolution_clock::now();
    // // __________________Create_Bit-Map_Character___________________
    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);

    // Step 1
    bitMapCreatorSimd<<<numBlock_8, BLOCKSIZE>>>( (uint32_t*) block_GPU, (uint8_t*) backslashes_GPU, (uint8_t*) quote_GPU, (uint8_t*) op_GPU, (uint8_t*) open_close_GPU, size, total_padded_8);
    cudaStreamSynchronize(0);



    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    // float milliseconds = 0;
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // std::cout << "Step 1 Time: " << milliseconds << " ms" << std::endl;

    // cout << "back_slash: \n";
    // print_d32(backslashes_GPU,total_padded_32,ROW1); 
    // cout << "dquote: \n";
    // print_d32(quote_GPU,total_padded_32,ROW1); 
    // cout << "op: \n";
    // print_d32(op_GPU,total_padded_32,ROW1); 
    // cout << "op: \n";
    // print_d32(open_close_GPU,total_padded_32,ROW1); 

    // __________________Find_Escaped_Character_____________________
    // Step 2
    uint32_t* real_quote_GPU = general_ptr + total_padded_32 * ROW4;
    // cudaEventRecord(start, 0);
    
    
    // findEscapedQuote<<<numBlock_8B, BLOCKSIZE>>>(backslashes_GPU, quote_GPU, real_quote_GPU, total_padded_32, total_padded_8B, WORDS);
    findEscapedQuoteMerge_NEW<<<numBlock_8B, BLOCKSIZE>>>(backslashes_GPU, quote_GPU, real_quote_GPU, total_padded_32, total_padded_8B, WORDS);
    cudaStreamSynchronize(0);

    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // std::cout << "Step 2 Time: " << milliseconds << " ms" << std::endl;
 
    // cout << "Time taken by program is Token Escaped Quote [step-2] : " << fixed << time_taken2 << setprecision(9);
    // cout << " sec" << endl;
    // print_d32(real_quote_GPU,total_padded_32,ROW1);
    // printf("findEscapedQuote Works Well!\n");


    // cout << "real quote: \n";
    // print_d32(real_quote_GPU,total_padded_32,ROW1); 
    // cout << "popc quote: \n";
    // print_d32(quote_GPU,total_padded_32,ROW1); 

    // Step 3a
    // __________________________REDUCE______________________________
     // Step 3a
    uint32_t* total_one_GPU = general_ptr;
    // uint32_t* total_one_32_GPU;
    // cudaMallocAsync(&total_one_32_GPU, (total_padded_32_div_32) * sizeof(uint32_t), 0);
    // cudaMallocAsync(&total_one_32_GPU, (total_padded_32_div_8) * sizeof(uint32_t), 0);
    // cudaEventRecord(start, 0);
    // reduceChunkBaseline<<<numBlock, BLOCKSIZE>>>(real_quote_GPU, total_one_GPU, total_padded_32);
    // reduceChunkBaseline64<<<numBlock_64, BLOCKSIZE>>>((uint64_t*) real_quote_GPU, (uint64_t*) total_one_GPU, total_padded_64);
    // reduceChunkBaseline4Words<<<numBlock_16B, BLOCKSIZE>>>(real_quote_GPU, total_one_GPU, total_padded_16B, size);
    // cudaStreamSynchronize(0);
    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // std::cout << "Step 3a Time: " << milliseconds << " ms" << std::endl;
    // cout << "popc quote: \n";
    // print_d32(total_one_GPU,total_padded_32,ROW1); 
    
    // Step 3b
    // cudaEventRecord(start, 0);
    // thrust::exclusive_scan(thrust::cuda::par, (uint64_t*) total_one_GPU, ( (uint64_t*) total_one_GPU ) + (total_padded_64), (uint64_t*) total_one_GPU);
    thrust::exclusive_scan(thrust::cuda::par, total_one_GPU, total_one_GPU + (total_padded_32), total_one_GPU);
    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // std::cout << "Step 3b Time: " << milliseconds << " ms" << std::endl;


    // cout << "scan popc quote: \n";
    // print_d32(total_one_GPU,total_padded_32,ROW1); 

    // Step 3d
    uint32_t* inString_GPU = general_ptr;

    // cudaEventRecord(start, 0);
    inStringFinderBaseline<<<numBlock, BLOCKSIZE>>>(real_quote_GPU, total_one_GPU, inString_GPU, total_padded_32);
    // inStringFinderBaseline64<<<numBlock_64, BLOCKSIZE>>>((uint64_t*) real_quote_GPU, (uint64_t*) total_one_GPU, (uint64_t*) inString_GPU, total_padded_64);
    cudaStreamSynchronize(0);
    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // std::cout << "Step 3d Time: " << milliseconds << " ms" << std::endl;

    // cout << "open close: \n";
    // print_d32(open_close_GPU,total_padded_32,ROW1); 


    // cout << "in string popc quote: \n";
    // print_d32(inString_GPU,total_padded_32,ROW1); 
    // exit(0);
    // Step 4
    // cudaEventRecord(start, 0);
    // findOutUsefulString<<<numBlock_8B, BLOCKSIZE>>>(op_GPU, newLine_GPU, inString_GPU, total_padded_32, total_padded_8B, WORDS);
    // cudaStreamSynchronize(0);
    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // std::cout << "Step 4 Time: " << milliseconds << " ms" << std::endl;

    // Step 4 merge with 5a
    uint32_t* set_bit_count = general_ptr + total_padded_32;
    // cudaEventRecord(start, 0);
    findOutUsefulStringMerge<<<numBlock_8B, BLOCKSIZE>>>(op_GPU, open_close_GPU, inString_GPU, total_padded_32, total_padded_8B, WORDS, set_bit_count);
    cudaStreamSynchronize(0);
    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // std::cout << "Step 4 Time: " << milliseconds << " ms" << std::endl;

    uint32_t* set_bit_count_open_close = op_GPU; // lets rename it for easy understanding
    uint32_t* structural_bitmap = inString_GPU;

    // cout << "out string quote: \n";
    // print_d32(inString_GPU,total_padded_32,ROW1); 
    // cout << "out string oc: \n";
    // print_d32(open_close_GPU,total_padded_32,ROW1); 
    // cout << "set bit counts quote: \n";
    // print_d32(set_bit_count,total_padded_32,ROW1); 
    // cout << "set bit counts open close: \n";
    // print_d32(set_bit_count_open_close,total_padded_32,ROW1); 
    // exit(0);

 

 
    // cout << "Time taken by program is Token [step-4] : " << fixed << time_taken7 << setprecision(9);
    // cout << " sec" << endl;
    // print_d(inString_GPU,total_padded_32,ROW1); 
    // printf("findOutUsefulString Works Well!\n");
    
    // ______________Final_Step_Write_____________________
    // step 5a:
    // uint32_t* set_bit_count = general_ptr + total_padded_32;
    // cudaEventRecord(start, 0);
    // count_set_bits<<<numBlock_8B, BLOCKSIZE>>>(structural_bitmap, set_bit_count, total_padded_32, total_padded_8B, WORDS);
    // cudaStreamSynchronize(0);
    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // std::cout << "Step 5a Time: " << milliseconds << " ms" << std::endl;

    // Step 5b
    // cudaEventRecord(start, 0);
    thrust::inclusive_scan(thrust::cuda::par, set_bit_count, set_bit_count + total_padded_32, set_bit_count);
    cudaMemcpyAsync(&last_index_tokens, set_bit_count + total_padded_32 - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // std::cout << "Step 5b Time: " << milliseconds << " ms" << std::endl;

    // Step 5c
    // cudaEventRecord(start, 0);
    thrust::inclusive_scan(thrust::cuda::par, set_bit_count_open_close, set_bit_count_open_close + total_padded_32, set_bit_count_open_close);
    cudaMemcpyAsync(&last_index_tokens_open_close, set_bit_count_open_close + total_padded_32 - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // std::cout << "Step 5c Time: " << milliseconds << " ms" << std::endl;


    // Step 5d
    // last_index_tokens += 3;
    // last_index_tokens_open_close += 2;

    int reminder = last_index_tokens % 4;    
    int padding = (4-reminder) & 3; 
    // It will always return a number between 0 and 3, 
    // which represents the number of padding bytes needed to align the size to the next multiple of 4.
    // uint64_t last_index_tokens_padded = (last_index_tokens + padding)/4;



    // uint8_t* out_string_8_GPU;
    uint32_t* out_string_8_index_GPU; // it's going to store real index.
    // cudaMallocAsync(&out_string_8_GPU, (last_index_tokens + padding) * sizeof(uint8_t),0);

    cudaMallocAsync(&out_string_8_index_GPU, last_index_tokens * sizeof(uint32_t) * ROW2,0); // Row 1 for structural index, Row 2 for ending pos which will calculated in parsr

    int reminder2 = last_index_tokens_open_close % 4;    
    int padding2 = (4-reminder2) & 3; 
    // It will always return a number between 0 and 3, 
    // which represents the number of padding bytes needed to align the size to the next multiple of 4.
    // uint64_t last_index_tokens_open_close_padded = (last_index_tokens_open_close + padding2)/4;



    uint8_t* out_string_open_close_8_GPU;
    uint32_t* out_string_open_close_8_index_GPU; // it's going to store structural index, not real index
    cudaMallocAsync(&out_string_open_close_8_GPU, (last_index_tokens_open_close + padding2)  * sizeof(uint8_t),0);
    cudaMallocAsync(&out_string_open_close_8_index_GPU, last_index_tokens_open_close * sizeof(uint32_t),0);

    // cout << "res size before remove copy: " << last_index_tokens_open_close << "\n";

    // cudaEventRecord(start, 0);
    removeCopy<<<numBlock, BLOCKSIZE>>>(set_bit_count,                      // prefix sum set bits until each word of structural
                                        set_bit_count_open_close,           // prefix sum set bits until each word of open close
                                        structural_bitmap,                  // structural bitmap out string
                                        open_close_GPU,                     // open close bitmap out string
                                        block_GPU,                          // real json block
                                        // out_string_8_GPU,                   // structural byte
                                        out_string_8_index_GPU,             // structural real index in real json file
                                        out_string_open_close_8_GPU,        // open_close byte
                                        out_string_open_close_8_index_GPU,  // structural index for each open close (not real json file)
                                        size, 
                                        last_index_tokens,                  // structural size
                                        last_index_tokens_open_close,       // open close size
                                        total_padded_32,
                                        lastStructuralIndex,                // last structural index from previous chunk
                                        lastChunkIndex);                    // last real json index from previous chunk
    cudaStreamSynchronize(0);
    
    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // std::cout << "Step 5d Time: " << milliseconds << " ms" << std::endl;

    // cudaMemcpyAsync(&last_index_tokens, set_bit_count+total_padded_32-1, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    // last_index_tokens += 3;
    cudaFreeAsync(general_ptr,0);


    in_string_out_index_d = out_string_8_index_GPU;
    // uint8_t* in_string_out_d;
    // in_string_out_d = out_string_8_GPU;
    ret_size = last_index_tokens; // latest index toye vagheait data

    open_close_d = out_string_open_close_8_GPU;
    open_close_index_d = out_string_open_close_8_index_GPU;

    // cout << "res size after remove copy: " << last_index_tokens_open_close << "\n";
    // print8_d<uint8_t>(in_string_out_d,ret_size,ROW1); 
    // exit(0);

    // cout << "res size: " << last_index_tokens_open_close << "\n";
    // print8_d<uint8_t>(open_close_d,last_index_tokens_open_close,ROW1); 
    // exit(0);

    // cudaEventRecord(start, 0);
    // cout << "index-after sort by key" << endl;
    // printUInt32ArrayFromGPU( open_close_index_d, last_index_tokens_open_close);

    // printf("removeCopy Works Well!\n");
    // cout << length << endl;
    // printInt32ArrayFromGPU(Row3Start, length);

    // exit(0);
    // cudaEventDestroy(start);
    // cudaEventDestroy(stop);

    return open_close_d;
}

__global__
void depth_init_MathAPI(uint32_t* open_close_GPU, uint32_t* oc_1, int oc_cnt_32, int oc_cnt){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int32_t i = index; i < oc_cnt_32 && i < oc_cnt ; i+=stride){
        uint32_t idx = i*4;
        uint32_t current_4_bytes = open_close_GPU[i];

        uint32_t isOpen = (__vcmpeq4(current_4_bytes, 0x5B5B5B5B) | __vcmpeq4(current_4_bytes, 0x7B7B7B7B) ) & 0x01010101; // 01
        uint32_t isClose = (__vcmpeq4(current_4_bytes, 0x5D5D5D5D) |  __vcmpeq4(current_4_bytes, 0x7D7D7D7D) );            // FF

        oc_1[i] = (isOpen |  isClose); 
    }
}

__global__
void validate_expand_MathAPI_new(char* structural_GPU, uint32_t* index_arr, uint32_t* endIdx, int oc_cnt_32, int oc_cnt, bool* error){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x; 
    // [ ] { } [ ] { }
    // 0 9 1 8 4 5 6 7
    __shared__ uint32_t shared_error;
    if(threadIdx.x == 0) shared_error = 0;
    __syncthreads();

    for(int32_t i = index; i < oc_cnt_32; i+=stride){
        int k = i*4; 
        int currentIndex = index_arr[k];

        if( i == oc_cnt_32 - 1){
            // printf("outside error: %c\n", structural_GPU[currentIndex]);
            if(k+1 >= oc_cnt){
                // printf("here error1: %c\n", structural_GPU[currentIndex]);
                atomicOr(&shared_error, 1); 
            }else if(k+2 >= oc_cnt){
                // printf("here error2: %c\n", structural_GPU[currentIndex]);
                int nextIndex = index_arr[k+1];
                uint32_t two_chars = structural_GPU[currentIndex] | structural_GPU[nextIndex] << 8;
                // [ ]
                // { }
                uint32_t error_local = (__vcmpeq2(two_chars, 0x5D5B) | __vcmpeq2(two_chars, 0x7D7B));
                atomicOr(&shared_error, ~error_local & 0x1); 
                endIdx[currentIndex] = nextIndex;
            }else if(k+3 >= oc_cnt){
                // printf("here error3: %c\n", structural_GPU[currentIndex]);
                atomicOr(&shared_error, 1); 
            }else{
                // printf("here error4: %c\n", structural_GPU[currentIndex]);
                int nextIndex = index_arr[k+1];
                int currentIndex_2 = index_arr[k+2];
                int nextIndex_2 = index_arr[k+3];
            
                uint32_t four_chars = structural_GPU[currentIndex] | structural_GPU[nextIndex] << 8 | structural_GPU[currentIndex_2] << 16 | structural_GPU[nextIndex_2] << 24;
                uint32_t shifted_four_chars = four_chars << 8;
                uint32_t xor_chars =  (four_chars ^ shifted_four_chars) & 0xFF00FF00;        
                uint32_t error_local = __vcmpeq4(xor_chars, 0x06000600);
                atomicOr(&shared_error, ~error_local > 0); 

                endIdx[currentIndex] = nextIndex;
                endIdx[currentIndex_2] = nextIndex_2;
            }   
        }else{    
            // printf("here error4: %c\n", structural_GPU[currentIndex]);
            int nextIndex = index_arr[k+1];

            int currentIndex_2 = index_arr[k+2];
            int nextIndex_2 = index_arr[k+3];
            
            // 5b xor 5d = 06
            // 7b xor 7d = 06
            uint32_t four_chars = structural_GPU[currentIndex] | structural_GPU[nextIndex] << 8 | structural_GPU[currentIndex_2] << 16 | structural_GPU[nextIndex_2] << 24;
            uint32_t shifted_four_chars = four_chars << 8;
            uint32_t xor_chars =  (four_chars ^ shifted_four_chars) & 0xFF00FF00;        
            uint32_t error_local = __vcmpeq4(xor_chars, 0x06000600) & 0xFFFFFFFF;

            atomicOr(&shared_error, ~error_local > 0); 

            endIdx[currentIndex] = nextIndex;
            endIdx[currentIndex_2] = nextIndex_2;
        }
        __syncthreads();
        if (threadIdx.x == 0 && shared_error) *error = true;
    }

}

__global__
void validate_expand_MathAPI_new2(char* pair_oc, uint32_t* index_arr, uint32_t* endIdx, int oc_cnt_32, int oc_cnt, bool* error, uint64_t lastStructuralIndex){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x; 
    // [ ] { } [ ] { }
    // 0 9 1 8 4 5 6 7
    __shared__ uint32_t shared_error;
    if(threadIdx.x == 0) shared_error = 0;
    __syncthreads();

    for(int32_t i = index; i < oc_cnt_32; i+=stride){
        int k = i*4; 
        // int currentIndex = index_arr[k];

        if( i == oc_cnt_32 - 1){
            // printf("outside error: %c\n", structural_GPU[currentIndex]);
            if(k+1 >= oc_cnt){
                // printf("here error1: %c\n", structural_GPU[currentIndex]);
                shared_error |= 1; 
            }else if(k+2 >= oc_cnt){
                // printf("here error1: %c\n", pair_oc[k]);
                // printf("here error2: %c\n", pair_oc[k+1]);
                uint32_t two_chars = pair_oc[k] | pair_oc[k+1] << 8;
                // printf("32: %x\n", two_chars);
                uint32_t shifted_two_char = two_chars << 8; 
                // printf("shifted_two_char 32: %x\n", shifted_two_char);
                uint32_t xor_chars =  (two_chars ^ shifted_two_char) & 0x0000FF00;        
                // printf("xor: %x\n", xor_chars);
                uint32_t error_local = __vcmpeq4(xor_chars, 0x00000600);
                // printf("err: %x\n", error_local);
                shared_error |= (~error_local) > 0; 

                endIdx[index_arr[k]] = index_arr[k+1] + lastStructuralIndex + 1;

            }else if(k+3 >= oc_cnt){
                // printf("here error3: %c\n", structural_GPU[currentIndex]);
                // atomicOr(&shared_error, 1); 
                shared_error |= 1; 
            }else{
                // printf("here error4: %c\n", structural_GPU[currentIndex]);
                // int nextIndex = index_arr[k+1];
                // int currentIndex_2 = index_arr[k+2];
                // int nextIndex_2 = index_arr[k+3];
            
                uint32_t four_chars = pair_oc[k] | pair_oc[k+1] << 8 | pair_oc[k+2] << 16 | pair_oc[k+3] << 24;
                uint32_t shifted_four_chars = four_chars << 8;
                uint32_t xor_chars =  (four_chars ^ shifted_four_chars) & 0xFF00FF00;        
                uint32_t error_local = __vcmpeq4(xor_chars, 0x06000600) & 0xFFFFFFFF;
                shared_error |= (~error_local) > 0; 

                endIdx[index_arr[k]] = index_arr[k+1] + lastStructuralIndex + 1;
                endIdx[index_arr[k+2]] = index_arr[k+3] + lastStructuralIndex + 1;
            }   
        }else{    
            // printf("here error4: %c\n", structural_GPU[currentIndex]);
            // int nextIndex = index_arr[k+1];

            // int currentIndex_2 = index_arr[k+2];
            // int nextIndex_2 = index_arr[k+3];
            
            // 5b xor 5d = 06
            // 7b xor 7d = 06
            
            uint32_t order_err = pair_oc[k] > pair_oc[k+1] | pair_oc[k+2] > pair_oc[k+3];
            uint32_t four_chars = pair_oc[k] | pair_oc[k+1] << 8 | pair_oc[k+2] << 16 | pair_oc[k+3] << 24;
            uint32_t shifted_four_chars = four_chars << 8;
            uint32_t xor_chars =  (four_chars ^ shifted_four_chars) & 0xFF00FF00;        
            uint32_t error_local = __vcmpeq4(xor_chars, 0x06000600) & 0xFFFFFFFF;

            shared_error |= (~error_local | order_err) > 0; 

            // atomicOr(&shared_error, (~error_local | order_err) > 0); 

            endIdx[index_arr[k]] = index_arr[k+1] + lastStructuralIndex + 1;
            endIdx[index_arr[k+2]] = index_arr[k+3] + lastStructuralIndex + 1;
        }
        __syncthreads();
        if (threadIdx.x == 0 && shared_error) *error = true;
    }

}



// int32_t* Parser(uint8_t* open_close_GPU, char* structural_GPU, int32_t** open_close_index_d,  int32_t** real_input_index_d, int oc_cnt, int structural_cnt, int & result_size) {
int32_t* Parser(uint8_t* open_close_GPU, int32_t** open_close_index_d,  int32_t** real_input_index_d, int oc_cnt, int structural_cnt, int & result_size, uint64_t lastStructuralIndex) {
    uint32_t* oc_idx = reinterpret_cast<uint32_t*>(*open_close_index_d);
    uint32_t* parsed_oc = reinterpret_cast<uint32_t*>(*real_input_index_d); // contains two rows--> 1. structural     2. pair_pos 

    
    // _______________STEP_1__(a)_________________    
    int numBlock = (structural_cnt + BLOCKSIZE - 1) / BLOCKSIZE;
    int numBlock_open_close = (oc_cnt + BLOCKSIZE - 1) / BLOCKSIZE;

    int WORDS = 4;
    int structural_cnt_32 = (structural_cnt + WORDS - 1) / WORDS;                   // for times that we are working on 4 bytes instead of 1 bytes in a thread
    int numBlock_32 = (structural_cnt_32 + BLOCKSIZE - 1) / BLOCKSIZE;

    int oc_cnt_32 = (oc_cnt + WORDS - 1) / WORDS;
    int numBlock_open_close_32 = (oc_cnt_32 + BLOCKSIZE - 1) / BLOCKSIZE;


    int32_t* res; // temporary result that will use in following

    uint32_t* oc_1; // output 
    cudaMallocAsync(&oc_1, oc_cnt_32*sizeof(uint32_t), 0); 

    depth_init_MathAPI<<<numBlock_open_close_32, BLOCKSIZE>>>( (uint32_t*) open_close_GPU, oc_1, oc_cnt_32, oc_cnt);
    cudaStreamSynchronize(0);


    uint32_t* depth = oc_1; // output 
    // // _______________STEP_1__(b)_________________
    thrust::inclusive_scan(thrust::cuda::par,  (uint8_t*) depth,  ((uint8_t*) depth) + oc_cnt,  (uint8_t*) depth); // on depth

    // // _______________STEP_2__(a)_________________
    thrust::transform_if(thrust::cuda::par, (uint8_t*) depth, ((uint8_t*) depth) + oc_cnt, open_close_GPU, (uint8_t*) depth, decrease(), is_opening());

    // // _______________STEP_3__(b)_________________

    // Use zip iterator to combine oc_idx and open_close_GPU
    auto zipped_begin = thrust::make_zip_iterator(thrust::make_tuple(oc_idx, open_close_GPU));
    thrust::stable_sort_by_key(thrust::cuda::par, (uint8_t*)depth, ((uint8_t*)depth) + oc_cnt, zipped_begin);

    char* pair_oc = (char *) open_close_GPU;
    uint32_t* pair_idx = oc_idx;


    // _______________STEP_4__(a)_________________
    // int error = 0;

    bool pairError = false;
    bool* pairError_GPU;
    cudaMallocAsync(&pairError_GPU, sizeof(bool), 0);                  //  Allocates Memory on the Device and Returns a Pointer to the Allocated Memory.
    cudaMemsetAsync(pairError_GPU, 0, sizeof(bool), 0);                //  Initializes a Block of Memory on the Device with a Specified Value
  
    uint32_t* pair_pos = parsed_oc + structural_cnt;
    // validate_expand_MathAPI_new<<<numBlock_open_close_32, BLOCKSIZE>>>(structural_GPU, pair_idx, end_pos, oc_cnt_32, oc_cnt, pairError_GPU); 
    validate_expand_MathAPI_new2<<<numBlock_open_close_32, BLOCKSIZE>>>(pair_oc, pair_idx, pair_pos, oc_cnt_32, oc_cnt, pairError_GPU, lastStructuralIndex); 

    cudaStreamSynchronize(0);
    cudaMemcpyAsync(&pairError, pairError_GPU, sizeof(bool), cudaMemcpyDeviceToHost, 0);

    if(pairError){  // 0 no error, 1 error
        printf("error found!");
        exit(0);
    }



    result_size = structural_cnt;


    cudaFreeAsync(open_close_GPU, 0);
    cudaFreeAsync(depth, 0);

    return (int32_t*) parsed_oc;
    //arr(output): ROW 1 depth (not anymore) | ROW1 Real Character Index | ROW2 End Index (for each opening)
}

cuJSONResult parse_standard_json(cuJSONInput input) {
    // initialize cuJSONoutput structure:
    cuJSONResult parsed_tree;
    parsed_tree.chunkCount = 1;
    parsed_tree.totalResultSize = 0;
    parsed_tree.resultSizes;
    parsed_tree.resultSizesPrefix;
    parsed_tree.structural = NULL;
    parsed_tree.pair_pos = NULL;

    int current_chunk_num = 0;
    int total_result_size = 0;          // latest index structural



    // Check if the input is valid
    if (input.data == nullptr || input.size == 0) {
        std::cerr << "\033[1;31m Error: Invalid JSON content or input.size. \033[0m\n";
        return cuJSONResult{};  // Return empty result
    }

                                         // fileSize
    size_t lastChunkIndex = 0;
    size_t lastStructuralIndex = 0;

    
    // init - Calculate padding to align the buffer size to the nearest multiple of 4 bytes for optimal GPU performance.
    int remainder = input.size % 4;    
    int padding = (4 - remainder) & 3;                                                                  // Padding bytes needed.
    uint64_t padded_length = input.size + padding;
    uint64_t size_32 = padded_length / 4;
    
    uint8_t* open_close_GPU;
    uint64_t * parse_tree; 
    
    // Host to Device Memory Copy - Allocate input memory on GPU
    uint8_t* d_jsonContent; // block_GPU
    cudaMalloc((void**)&d_jsonContent, (input.size + padding) * sizeof(uint8_t));
    cudaMemset(d_jsonContent, 0, (input.size + padding) * sizeof(uint8_t));
    cudaMemcpy(d_jsonContent, input.data, input.size * sizeof(uint8_t), cudaMemcpyHostToDevice);


    // UTF8 Validation
    bool isValidUTF8 = UTF8Validation(reinterpret_cast<uint32_t *>(d_jsonContent), size_32);
    cudaStreamSynchronize(0);
    if(!isValidUTF8) {
        exit(0);
    }


    // Tokenization
    uint32_t last_index_tokens;
    uint32_t last_index_tokens_open_close;
    int ret_size = 0;
    uint32_t* tokens_index_GPU;
    uint32_t* open_close_index_GPU;
    open_close_GPU = Tokenize(d_jsonContent, input.size, ret_size, last_index_tokens, last_index_tokens_open_close, tokens_index_GPU, open_close_index_GPU, lastStructuralIndex, lastChunkIndex);


    // Structure Recognition
    int32_t* result_GPU;
    int32_t* result;
    int result_size;
    result_GPU = Parser(open_close_GPU, 
                        (int32_t **)(&open_close_index_GPU), 
                        (int32_t **)(&tokens_index_GPU), 
                        last_index_tokens_open_close, 
                        last_index_tokens, 
                        result_size,
                        lastStructuralIndex);



    uint32_t total_tokens = (uint32_t) last_index_tokens;

    // output_size = (uint32_t) result_size * ROW2;
    cudaFree(d_jsonContent); // Free the input memory on GPU


    // Device to Host Memory Copy
    // int32_t* res_buf_arrays[1];
    // res_buf_arrays[0] = (int32_t*) malloc(sizeof(int32_t) * result_size * ROW2);
    // cudaMemcpy(1 + res_buf_arrays[0], result_GPU, sizeof(int32_t) * result_size, cudaMemcpyDeviceToHost); // first and last is for [ and ]
    // cudaMemcpy(1 + res_buf_arrays[0] + 1 + result_size,   result_GPU + result_size, sizeof(int32_t) * result_size, cudaMemcpyDeviceToHost); // first and last is for [ and ]
    int32_t* res_buff;
    size_t res_buff_count = (size_t)(result_size + 2) * ROW2;
    cudaMallocHost((void**)&res_buff, sizeof(int32_t) * res_buff_count);  // Pinned memory for fast H2D copy
    memset(res_buff, 0, sizeof(int32_t) * res_buff_count);

    // Copy results from device to host
    // res_buff[0] and res_buff[result_size + 1] are reserved for artificial
    // root/sentinel entries. Device structural data starts at structural[1],
    // and device pair_pos data starts at pair_pos[1].
    cudaMemcpy(1 + res_buff, result_GPU, sizeof(int32_t) * result_size, cudaMemcpyDeviceToHost);  // result 1
    cudaMemcpy(1 + res_buff + 1 + result_size, result_GPU + result_size, sizeof(int32_t) * result_size, cudaMemcpyDeviceToHost);  // result 2

    
    total_result_size += result_size;
    parsed_tree.resultSizesPrefix.push_back(total_result_size);
    parsed_tree.resultSizes.push_back(result_size);

    // parsed tree struct generation
    parsed_tree.totalResultSize = total_result_size + 2;
    parsed_tree.fileSize = result_size + 2;
    
    // Initialize artificial root/sentinel entries now, so parsed_tree is valid
    // immediately after parse_standard_json() returns.
    res_buff[0] = 0;
    res_buff[result_size + 1] = parsed_tree.totalResultSize - 1;
    
    // parsed_tree.structural = res_buf_arrays[0];
    // parsed_tree.pair_pos = res_buf_arrays[0] + total_result_size + 1;
    parsed_tree.structural = res_buff;
    parsed_tree.pair_pos = res_buff + result_size + 1;  // skip structural section


    // cout << "Total Result Size = " << parsed_tree.totalResultSize << endl;
    // cout << "File Size = " << parsed_tree.fileSize << endl;
    cudaFree(input.data);
    return parsed_tree;
}

