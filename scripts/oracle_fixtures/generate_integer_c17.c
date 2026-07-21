/* Independent C17 generator for the CP-INTEGER bootstrap fixture.
 *
 * It prints canonical JSON to stdout and deliberately contains no TorchCTS or
 * PyTorch dependency.  Inputs stay within the defined range for conversion to
 * int64_t; conversion to unsigned types then follows C17 modulo semantics.
 */

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

static uint64_t unsigned_magnitude(int64_t value) {
    return value < 0 ? UINT64_C(0) - (uint64_t)value : (uint64_t)value;
}

static uint64_t gcd_u64(int64_t left, int64_t right) {
    uint64_t a = unsigned_magnitude(left);
    uint64_t b = unsigned_magnitude(right);
    while (b != 0) {
        uint64_t remainder = a % b;
        a = b;
        b = remainder;
    }
    return a;
}

int main(void) {
    static const double byte_inputs[] = {
        -513.0, -256.0, -255.0, -1.0, -0.9, 0.0, 0.9, 1.0, 255.0, 256.0, 511.0
    };
    static const uint8_t u8_inputs[] = {0, 1, UINT8_MAX, UINT8_C(0x80)};
    static const uint16_t u16_inputs[] = {0, 1, UINT16_MAX, UINT16_C(0x8000)};
    static const uint32_t u32_inputs[] = {0, 1, UINT32_MAX, UINT32_C(0x80000000)};
    static const uint64_t u64_inputs[] = {0, 1, UINT64_MAX, UINT64_C(0x8000000000000000)};
    static const int64_t gcd_inputs[][2] = {
        {INT64_MIN, 31}, {INT64_MIN, -31}, {INT64_MIN, 0}, {84, -30}
    };
    size_t index;

    fputs("{\"float_to_uint8\":[", stdout);
    for (index = 0; index < sizeof(byte_inputs) / sizeof(byte_inputs[0]); ++index) {
        uint8_t result = (uint8_t)(int64_t)byte_inputs[index];
        printf("%s%" PRIu8, index ? "," : "", result);
    }
    fputs("],\"gcd_int64\":[", stdout);
    for (index = 0; index < sizeof(gcd_inputs) / sizeof(gcd_inputs[0]); ++index) {
        printf("%s%" PRIu64, index ? "," : "", gcd_u64(gcd_inputs[index][0], gcd_inputs[index][1]));
    }
    fputs("],\"unsigned_negation\":{\"uint8\":[", stdout);
    for (index = 0; index < sizeof(u8_inputs) / sizeof(u8_inputs[0]); ++index) {
        printf("%s%" PRIu8, index ? "," : "", (uint8_t)(-u8_inputs[index]));
    }
    fputs("],\"uint16\":[", stdout);
    for (index = 0; index < sizeof(u16_inputs) / sizeof(u16_inputs[0]); ++index) {
        printf("%s%" PRIu16, index ? "," : "", (uint16_t)(-u16_inputs[index]));
    }
    fputs("],\"uint32\":[", stdout);
    for (index = 0; index < sizeof(u32_inputs) / sizeof(u32_inputs[0]); ++index) {
        printf("%s%" PRIu32, index ? "," : "", (uint32_t)(-u32_inputs[index]));
    }
    fputs("],\"uint64\":[", stdout);
    for (index = 0; index < sizeof(u64_inputs) / sizeof(u64_inputs[0]); ++index) {
        printf("%s%" PRIu64, index ? "," : "", (uint64_t)(-u64_inputs[index]));
    }
    fputs("]}}\n", stdout);
    return 0;
}
