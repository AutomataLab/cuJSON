#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iomanip>
#include <x86intrin.h>
#include <string>
#include <vector>
#include <queue>
#include <inttypes.h>
#include <sys/resource.h>
#include <stdint.h>
#include <algorithm>
#include "../cujson_types.h"

using namespace std;

// all over this code, "idx" and "node" will use for indexes of structural array. 
//                   , "pos" will use for indexes of real json file.
class cuJSONLinesIterator{
    public:
        uint8_t* inputJSON = nullptr;                  // JSON string
        vector<int> resultSizes;
        vector<int> resultSizesPrefix;
        int32_t* structural;
        int32_t* pair_pos;

        int jsonDepth;
        int bufferSize;
        int totalResultSize;

        int chunkCount;
        int currentChunkIndex;

        int fileSize;

        

        token_type nodeType = OBJECT;
        int node = 0;
        int currArrayNode = 0;
        // int node_depth = 1;

        // main functions
        std::string getKey();                           // return the key of current node (idx)
        std::string getValue();                         // return the value of current node (idx)
        int findKey(string key);                        // this function will change the pointer of iterator to the colon that has the specific key
        int gotoKey(string key);                        // goto colon of specific key
        int gotoArrayIndex(int index);                  // goto [ or comma of specific index within an Arrat
        int increamentIndex(int index);                 // go forward for index size
        int gotoNextSibling(int index);                 // go forward to next sibling in an array
        bool checkKeyValue(string key, string value);   // check if the current object has the key-value pair

        // helper functions
        char getChar(int structuralIdx);                // get real json characters according to the node idx
        char getArtificalChar(int idx);
        void reset();                                   // reset pointer iterator to the first idx of structural
        void freeJson();
        int gotoArrayIndexSiblingHelper(int index);    

        
        // read file function
        bool readFile(const char* file, uint8_t*& buffer, size_t& size);



        // jump functions
        int jumpOpeningForward(int idx);                // jump from opening node idx to its corresponding closing idx
        int jumpSpacesForward(int pos);                 // jump over all spaces from current node pos to first non-space pos forward
        int jumpSpacesBackward(int pos);                // jump over all spaces from current node pos to first non-space pos backward
        // int jumpBackward(int pos);                      // jump over all spaces and closing from current node pos to first non-space and non-closing pos backward
        int jumpForwardStructural(int idx);

        int jumpValueBackward(int pos);
        int jumpValueForward(int pos);
        // future functions
        // token_type getType();
        // bool has_key();
        // bool has_value();

        size_t len = 0;
        

    cuJSONLinesIterator(cuJSONResult* parsedTree, const char* filePath){
        if(!readFile(filePath, inputJSON, len)){
            std::cout << "\033[1;31m[ERR]\033[0m File loading for the query iterator failed. Please check the file path.\n";
        }
        // inputJSON = parsedTree.inputJSON;                  // JSON string
        resultSizes = parsedTree->resultSizes;
        resultSizesPrefix = parsedTree->resultSizesPrefix;
        totalResultSize = parsedTree->totalResultSize;

        
        structural = parsedTree->structural;
        structural[0] = 0;

        fileSize = parsedTree->fileSize;
        structural[totalResultSize-1] = fileSize - 1;
        // loop over the structrual array and print the inputJSON in structural array value
        // for(int i = 1; i < totalResultSize && i < 100; i++){
        //     cout << "structural[" << i << "] = " << structural[i] << "\t" << inputJSON[structural[i] - 1] << endl;
        // }   

        pair_pos = parsedTree->pair_pos;
        pair_pos[0] = totalResultSize-1;

        jsonDepth = parsedTree->depth;
        bufferSize = parsedTree->bufferSize;

        chunkCount = parsedTree-> chunkCount;
        currentChunkIndex = 0;
    }

    private: // for printing we always print it from startIdx to endIdx
        string getString(int start_index, int end_index);   // use for getting key
        string getStringBackward(int end_index);
        string getValueBackward(int start_index, int end_index, primitive_type& type);
        int getValue(int index);
}; 


void cuJSONLinesIterator::freeJson(){
    free(inputJSON);
    cudaFreeHost(structural);
}

char cuJSONLinesIterator::getChar(int idx){
    // Fixed: Check if idx is within bounds and structural/inputJSON are not NULL
    // Also, make sure pos is within the bounds of the inputJSON array
    if(idx < 0 || idx >= totalResultSize || structural == NULL || inputJSON == NULL){
        return '\0';
    }
    else if (idx == totalResultSize - 1) return ']';
    else if (idx == 0) return '[';

    int pos = structural[idx] - 1;
    if(pos < 0 || pos >= fileSize){
        return '\0';
    }
    else if(inputJSON[pos] == '\n'){
        return ',';
    }
    else return inputJSON[pos];
}

int cuJSONLinesIterator::jumpOpeningForward(int idx){
    // Fixed: Check if idx is within bounds and pair_pos is not NULL
    if(idx < 0 || idx >= totalResultSize || pair_pos == NULL){
        return -1;
    }

    int pairNode = pair_pos[idx];
    // Fixed: Check if pairNode is within bounds
    if(pairNode < 0 || pairNode >= totalResultSize){
        return -1;
    }

    return pairNode;
}

int cuJSONLinesIterator::jumpSpacesForward(int pos){
    int current_pos = pos;
    while(inputJSON[current_pos] == ' '){
        current_pos++;
    }
    return current_pos; // first place that is not space
}

int cuJSONLinesIterator::jumpSpacesBackward(int pos){
    int current_pos = pos;
    while(inputJSON[current_pos] == ' '){
        current_pos--;
    }
    return current_pos; // last place that is not space
}

int cuJSONLinesIterator::jumpValueBackward(int pos){
    int current_pos = pos - 1;                                                          // pass its current char
    while(inputJSON[current_pos] == ' ' || inputJSON[current_pos] == '"'){
        if(inputJSON[current_pos] == '"'){
            return current_pos-1;                                                       // pass double-quote
        }
        current_pos--;                                                                  // passing spaces
    }
    return current_pos;                                                                 // last place that is not space
}

int cuJSONLinesIterator::jumpValueForward(int pos){
    int current_pos = pos + 1;                                                          // pass its current char
    while(inputJSON[current_pos] == ' ' || inputJSON[current_pos] == '"'){
        if(inputJSON[current_pos] == '"'){
            return current_pos+1;                                                       // pass double-quote
        }
        current_pos++;                                                                  // passing spaces
    }
    return current_pos;                                                                 // last place that is not space
}

char cuJSONLinesIterator::getArtificalChar(int idx){
    if(idx == totalResultSize - 1) return ']';
    else return NULL;
}

int cuJSONLinesIterator::jumpForwardStructural(int idx){
    int current_idx = idx + 1;
    char current = getChar(current_idx);
    while(current == '}' || current == ']' || current == getArtificalChar(current_idx)){
        current_idx++;
        current = getChar(current_idx);
    }
    return current_idx; // last place that is not space
}

bool cuJSONLinesIterator::readFile(const char* file, uint8_t*& buffer, size_t& size) {
    FILE* handle = fopen(file, "rb");
    if (!handle) {
        std::cerr << "File not found!" << std::endl;
        return false;
    }

    // Determine the size of the file
    fseek(handle, 0, SEEK_END);
    size = ftell(handle);
    fseek(handle, 0, SEEK_SET);

    // Allocate memory for the buffer
    buffer = new uint8_t[size];
    if (!buffer) {
        std::cerr << "Memory allocation failed!" << std::endl;
        fclose(handle);
        return false;
    }

    // Read the file into the buffer
    size_t bytesRead = fread(buffer, 1, size, handle);
    if (bytesRead != size) {
        std::cerr << "Reading error!" << std::endl;
        delete[] buffer;
        fclose(handle);
        return false;
    }

    fclose(handle);
    return true;
}

// in order to reset the pointer to the first real position
void cuJSONLinesIterator::reset(){
    node = 0;
    currArrayNode = 0;
    // node_depth = 1;
    nodeType = OBJECT;
}

int cuJSONLinesIterator::gotoArrayIndex(int index){
    // Fixed: Check if index is valid and structural/inputJSON are not NULL
    if(index < 0){
        return 0;
    }

    if(totalResultSize <= 0 || structural == NULL || pair_pos == NULL || inputJSON == NULL){
        return 0;
    }

    if(node < 0 || node >= totalResultSize){
        return 0;
    }

    int total = index + 1;                                // total number of index that we have to go forward to get the requested index [started from 0]
                                                          // +1 is for handling indexes [1,2,3,...], user will use [0,1,2,...]
    int startNode = node;
    int nextNode;
    int newCurrArrayNode = currArrayNode;

    char currentNodeChar = getChar(startNode);
    // Fixed: Check if currentNodeChar is valid
    if(currentNodeChar == '\0'){
        return 0;
    }

    /*
     * Do not call increamentIndex(1) here because it mutates iterator state
     * before we know whether the requested array index is valid.
     */
    if(currentNodeChar == ',' || currentNodeChar == '\n' || currentNodeChar == ':'){
        // Fixed: Check if startNode + 1 is within bounds
        if(startNode + 1 >= totalResultSize){
            return 0;
        }

        startNode++;
        currentNodeChar = getChar(startNode);
        // Fixed: Check if currentNodeChar is valid
        if(currentNodeChar == '\0'){
            return 0;
        }
    }

    // next node
    nextNode = startNode + 1;
    // Fixed: Check if nextNode is within bounds
    if(nextNode < 0 || nextNode >= totalResultSize){
        return 0;
    }
    char nextNodeChar = getChar(nextNode);
    // Fixed: Check if nextNodeChar is valid
    if(nextNodeChar == '\0'){
        return 0;
    }
    // cout << "nextNodeChar: " << nextNodeChar <<endl; 

    // total != 1 because we consider '[' as node to first index in array
    while( total != 1 && nextNodeChar != ']' && nextNode != totalResultSize - 1){ 
        // Fixed: Check if nextNode is within bounds  
        if(nextNode < 0 || nextNode >= totalResultSize){
            return 0;
        }
        // cout << "nxt->" << nextNodeChar <<endl;     
        if(nextNodeChar == '[' || nextNodeChar == '{'){
            // cout << "nextNodeChar: " << nextNodeChar << endl;
            int pairNode = jumpOpeningForward(nextNode);
            // Fixed: Check if pairNode is within bounds
            if(pairNode <= nextNode || pairNode >= totalResultSize){
                return 0;
            }

            nextNode = pairNode;
        }
        if(nextNodeChar == ',' || nextNodeChar == '\n' || nextNodeChar == ' '){ // no need for \n
            newCurrArrayNode  = nextNode; // save the current array node
            total--; // go one node forward
        }

        nextNode++;
        // Fixed: Check if nextNode is within bounds
        if(nextNode < 0 || nextNode >= totalResultSize){
            return 0;
        }
        nextNodeChar = getChar(nextNode);
        // Fixed: Check if nextNodeChar is valid
        if(nextNodeChar == '\0'){
            return 0;
        }
    }
   
    // cout << "total: " << total << endl;
    // cout << "nextNode: " << nextNode << endl;
    // cout << "node: " << node << endl;
    // cout << "current array node: " << currArrayNode << endl;
    // // char charrrr = getchar(currArrayNode);
    // cout << "current array char node: " << charrrr  << endl;
    // that means we achieve to requested index
    if(total == 1){
        // cout << "curr node in total == 1 --> " << getChar(nextNode) <<endl;
        int targetNode = nextNode - 1;
        // Fixed: Check if targetNode is within bounds
        if(targetNode < 0 || targetNode >= totalResultSize){
            return 0;
        }
        /*
         * Reject empty-array access, for example gotoArrayIndex(0) on [].
         */
        if(nextNodeChar == ']' && targetNode == startNode){
            return 0;
        }
        node = targetNode; // change the node pointer iterator to the nextNode pointer
        currArrayNode = newCurrArrayNode;

        // cout << "node - 2: " << node << endl;
        // currentNodeChar = getChar(node);
        // cout << "currentNodeChar-2: " << currentNodeChar <<endl;

        // if (node != 0){
        //     char previousNodeChar = getChar(node-1); // save the previous node char
        //     cout << "previousNodeChar-2: " << previousNodeChar <<endl;
        // }
        // char nextNodeChar = getChar(nextNode); // save the next node char
        // cout << "nextNodeChar-2: " << nextNodeChar <<endl;
        // currArrayNode = node;

        // cout << "node: " << node <<endl; 
        if(nextNodeChar == '{'){
            nodeType = OBJECT;
        }
        else if(nextNodeChar == '['){
            nodeType = ARRAY;
        }
        else if(nextNodeChar == ',' || currentNodeChar == '\n' || nextNodeChar == ']'){
            nodeType = VALUE;
        }
        return node; // node
    }
    // if(toto)
    return 0; // error
}

int cuJSONLinesIterator::gotoArrayIndexSiblingHelper(int index){
    int total = index + 1;                                // total number of index that we have to go forward to get the requested index [started from 0]
                                                          // +1 is for handling indexes [1,2,3,...], user will use [0,1,2,...]
    
    char currentNodeChar = getChar(node);
    // cout << "currNodeChar: " << currentNodeChar <<endl; 
    // if(currentNodeChar == ',' || currentNodeChar == '\n' || currentNodeChar == ':') increamentIndex(1);
    // next node
    int nextNode = node+1;         
    char nextNodeChar = getChar(nextNode);

    // cout << "nextNodeChar: " << nextNodeChar <<endl; 

    // total != 1 because we consider '[' as node to first index in array
    while( total != 1 && nextNodeChar != ']' && nextNode != totalResultSize - 1){   
        // cout << "nxt->" << nextNodeChar <<endl;     
        if(nextNodeChar == '[' || nextNodeChar == '{'){
            // cout << "nextNodeChar: " << nextNodeChar << endl;
            nextNode = jumpOpeningForward(nextNode);

        }
        if(nextNodeChar == ',' || nextNodeChar == '\n' || nextNodeChar == ' '){ // no need for \n
            total--; // go one node forward
        }

        nextNode++;
        nextNodeChar = getChar(nextNode);
    }
   
    // cout << "total: " << total << endl;
    // cout << "nextNode: " << nextNode << endl;
    // cout << "node: " << node << endl;
    // cout << "current array node: " << currArrayNode << endl;
    // // char charrrr = getchar(currArrayNode);
    // cout << "current array char node: " << charrrr  << endl;
    // that means we achieve to requested index
    if(total == 1){
        // cout << "curr node in total == 1 --> " << getChar(nextNode) <<endl;
        node = nextNode-1; // change the node pointer iterator to the nextNode pointer
        currArrayNode = node; // save the current array node
        // cout << "node - 2: " << node << endl;
        // currentNodeChar = getChar(node);
        // cout << "currentNodeChar-2: " << currentNodeChar <<endl;

        // if (node != 0){
        //     char previousNodeChar = getChar(node-1); // save the previous node char
        //     cout << "previousNodeChar-2: " << previousNodeChar <<endl;
        // }
        // char nextNodeChar = getChar(nextNode); // save the next node char
        // cout << "nextNodeChar-2: " << nextNodeChar <<endl;
        // currArrayNode = node;

        // cout << "node: " << node <<endl; 
        if(nextNodeChar == '{'){
            nodeType = OBJECT;
        }
        else if(nextNodeChar == '['){
            nodeType = ARRAY;
        }
        else if(nextNodeChar == ',' || currentNodeChar == '\n' || nextNodeChar == ']'){
            nodeType = VALUE;
        }
        return node; // node
    }
    // if(toto)
    return 0; // error
}


int cuJSONLinesIterator::gotoNextSibling(int index){
    // cout << "go to next sibling: " << node << endl;
    node = currArrayNode;
    // cout << "node: " << node << endl;
    // char currentNodeChar = getChar(node);
    // cout << "currentNodeChar: " << currentNodeChar <<endl;
    
    
    int idx = gotoArrayIndexSiblingHelper(index); // go to next sibling in array
    // if(idx == 0)
    //     return node; 
    // else
    //     return idx; // return the next node index
    return idx;
}

int cuJSONLinesIterator::increamentIndex(int index){
    node = node + index;
    char currentNodeChar = getChar(node);

    if(currentNodeChar == '{'){
        nodeType = OBJECT;
    }else if(currentNodeChar == '['){
        nodeType = ARRAY;
    }else if(currentNodeChar == ',' || currentNodeChar == '\n'){
        nodeType = VALUE;
    }else if(currentNodeChar == ':'){
        nodeType = KEYVALUE;
    }else if(currentNodeChar == ']' || currentNodeChar == '}'){
        // error
        nodeType = CLOSING;
    }else{
        return 0;
    }


    // cout << "goto index char: " << currentNodeChar << endl;
    // if(currentNodeChar == ':'){
    //     node++;
    //     currentNodeChar = getChar(node);
    // }
    // if(currentNodeChar == ']' || currentNodeChar == '}'){
    //     node--;
    //     currentNodeChar = getChar(node);
    // }
    

    // printf("%d current char %c\n", node, currentNodeChar);
    // if(currentNodeChar == '{'){
    //     nodeType = OBJECT;
    // }else if(currentNodeChar == '['){
    //     nodeType = ARRAY;
    // }else if(currentNodeChar == ',' || currentNodeChar == '\n'){
    //     nodeType = VALUE;
    // }else if(currentNodeChar == ':'){
    //     nodeType = KEYVALUE;
    // }else if(currentNodeChar == ']' || currentNodeChar == '}'){
    //     node++;
    //     nodeType = CLOSING;
    // }else{
    //     return 0;
    // }
    return node;               // number of movement for that index
}

string cuJSONLinesIterator::getString(int startPos, int endPos){ // it will use for getKeys and end string is always comma
    int i = 0;
    int length = 0;
    string result;

    startPos = jumpSpacesForward(startPos+1);
    endPos = jumpSpacesBackward(endPos-1);

    length = endPos - startPos - 1;                           
    result.assign((char*)(inputJSON+startPos+1), abs(length)); // get string based on opening and length
    return result;
}

int cuJSONLinesIterator::findKey(string input_key){
    char currentNodeChar = getChar(node);
    int nextNode = node;

    if(currentNodeChar == ':' || currentNodeChar == ',' || currentNodeChar == '\n'){ // it might be part of a VALUE or a KEYVALUE.
        nextNode = nextNode + 1;
    }

    char nextPossibleNodeChar = getChar(node+1);
    if( currentNodeChar == '[' && nextPossibleNodeChar == '{'){
        currentNodeChar = nextPossibleNodeChar;
        nextNode = node + 1;
    }

    if(getChar(nextNode) != '{'){
        // cout << "ERROR: Node is not an object to find key!" << endl;
        return 0;
    }

    int endNode = jumpOpeningForward(nextNode);          // to have the checking range of this depth
    nextNode = nextNode+1; 
    
    char nextNodeChar = getChar(nextNode);
    while (nextNode < endNode && nextNodeChar != '}')
    {
        if(nextNodeChar == '[' || nextNodeChar == '{'){ //opening
            nextNode = jumpOpeningForward(nextNode);
        }                    
        if(nextNodeChar == ':'){                    // check key
            string key;
            int endPos = structural[nextNode] - 1;
            int startIdx = nextNode-1;
            int startPos = structural[startIdx]-1;
            key = getString(startPos, endPos);
            if(key.compare(input_key)==0){
                return nextNode - node;             // return the idx : will give us the idx of that key (comma before it)
            }

        }

        nextNode++;
        nextNodeChar = getChar(nextNode);
    }
    return 0;
}

string cuJSONLinesIterator::getKey(){
    char currentNodeChar = getChar(node);
    // cout << "currentNodeGetkey: " << currentNodeChar <<endl;
    string key;

    if(currentNodeChar == ':'){
        int endIdx = node; 
        int endPos = structural[endIdx] - 1;
        int startIdx = node - 1;
        int startPos = structural[startIdx]-1;
        key = getString(startPos, endPos); // key = getString(key_start_char_index+1, key_char_index);
        // cout << "key---->" << key << endl;
        return key;
    }else{
        cout << "ERROR! The iterator must point to a colon.";
        exit(0);
    }
}

string cuJSONLinesIterator::getValue(){
    // cout << "getValue called for node: " << node << endl;
    int startPos = structural[node] - 1;
    // cout << "startPos: " << startPos << endl;
    char currentNodeChar = getChar(node);
    string value;
    primitive_type type;
    if(currentNodeChar == ',' || currentNodeChar == '\n' || currentNodeChar == ':'){
        int startIdx = node + 1;            // maybe its array
        int startPos, endIdx, endPos;

        int nextNodeChar = getChar(startIdx);
        if(nextNodeChar == '[' || nextNodeChar == '{'){ // returning array or object as value
            endIdx = jumpOpeningForward(startIdx);
            startPos = structural[startIdx] - 1;
            endPos = structural[endIdx]-1;
            value = string((char*) inputJSON + startPos, endPos - startPos + 1);
            return value;
        }else{
            endIdx = startIdx;
            startIdx = node;

            startPos = jumpValueForward ( structural[startIdx] - 1);
            endPos = jumpValueBackward  ( structural[endIdx]   - 1);
            value = string((char*) inputJSON + startPos, endPos - startPos + 1);
            return value;
        }
    }else if(currentNodeChar == '['){
        int startIdx = node + 1;            // maybe its array
        int startPos, endIdx, endPos;

        int nextNodeChar = getChar(startIdx);
        if(nextNodeChar == '[' || nextNodeChar == '{'){ // returning array or object as value
            endIdx = jumpOpeningForward(startIdx);

            startPos = structural[startIdx] - 1;
            endPos = structural[endIdx] - 1;
            value = string((char*) inputJSON + startPos, endPos - startPos + 1);
            return value;
        }else{
            endIdx = startIdx;
            startIdx = node;

            startPos = jumpValueForward ( structural[startIdx] - 1);
            endPos = jumpValueBackward  ( structural[endIdx]   - 1);
            value = string((char*) inputJSON + startPos, endPos - startPos + 1);
            return value;
        }
    }else{
        cout << 'ERROR: iterator is in wrong place.' <<endl;
    }

    // string value;
    // primitive_type type;
    // // printf("value char %c\n", currentNodeChar);
    // if(currentNodeChar == '[' || currentNodeChar == '{'){ // object or array
    //     int endIdx = jumpOpeningForward(node);
    //     int endPos = structural[endIdx]-1;
    //     value = string((char*) inputJSON + startPos, endPos - startPos + 1);
    //     // cout << "vlaue before return: " << value << endl;
    //     return value;
    // }
    // else if(currentNodeChar == ':'){ // value
    //     int endIdx = node + 1;
    //     int endPos = structural[endIdx]-1;
    //     int end_char = getChar(endIdx);
    //     // printf("start: %d, end: %d\n", char_index, end_char_index);

    //     if(end_char == ',' || end_char == ']' || end_char == '}' || end_char == '\n'){
    //         endPos = structural[endIdx]-1;
    //         // printf("start: %c, end: %c\n", inputJSON[char_index+1], inputJSON[end_char_index-1]);
    //         value = getValueBackward(startPos, endPos, type);
    //         // value = getValueBackward(startIdx+1, endIdx, type);
    //     }
    //     else if(end_char == '[' || end_char == '{'){
    //         int startIdx = endIdx;
    //         startPos = structural[startIdx] - 1;

    //         endIdx = endIdx + 1; // node + 2;
    //         endPos = structural[endIdx] - 1;
    //         // printf("start: %c, end: %c\n", inputJSON[start_char_index], inputJSON[end_char_index]);
    //         value = getValueBackward(startPos, endPos, type);
    //         // value = getValueBackward(start_char_index+1, end_char_index-1, type);
    //     }
    //     return value;
    // }
    // else if(currentNodeChar == ',' || currentNodeChar == '\n'){
    //     int endPos = startPos;
    //     int startIdx = node - 1;
    //     startPos = structural[startIdx]-1; 
    //     value = getValueBackward(startPos, endPos, type); // no -1 or +1
    //     // value = getValueBackward(start_char_index+1, char_index-1, type);
    //     return value;
    // }
    // return value;
} // we have to change the place of idx and pos with each other

int cuJSONLinesIterator::gotoKey(string key){
    int index1 = findKey(key);
    return increamentIndex(index1);
}



bool cuJSONLinesIterator::checkKeyValue(string key, string value){
    int index1 = findKey(key);
    if(index1 == 0){
        return false; // key not found
    }
    int currentNode = node;
    increamentIndex(index1); // go to the colon of the key
    string currentValue = getValue(); // get the value of the key
    if(currentValue.compare(value) == 0){
        node = currentNode; // reset the node pointer to the previous position
        // cout << "key: " << key << ", value: " << currentValue << endl;
        return true; // key-value pair found
    }else{
        node = currentNode; // reset the node pointer to the previous position
        return false;
    }
}