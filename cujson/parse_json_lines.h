#ifndef STANDARD_PARSE_H
#define STANDARD_PARSE_H

#include <string>
#include "cujson_types.h"

// Function prototype for standard_parse
// cuJSONResult standard_json_parse(uint8_t* h_jsonContent);
cuJSONResult parse_json_lines(cuJSONLinesInput input);

#endif // STANDARD_PARSE_H