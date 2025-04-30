#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

#define MAX_DEPTH 27

__kernel void buildSignature(
    __global const char* boards,
    __global unsigned long* out,
    const int step)
{
    int gid = get_global_id(0);
    unsigned long x = 0UL;
    unsigned long o = 0UL;
    int base = gid * MAX_DEPTH;

    for (int i = 0;  i < step;  i++) {
        char c = boards[base + i];
        if (c >= 'A' && c <= 'Z') {
            int idx = c - 'A';
            x |= (1UL << idx);
        }
        else if (c >= 'a' && c <= 'z') {
            int idx = c - 'a';
            o |= (1UL << idx);
        }
        else if (c == '.') {
            // board cell for X
            x |= (1UL << 26);
        }
        else if (c == ',') {
            // board cell for O
            o |= (1UL << 26);
        }
    }
    // pack X in upper 27 bits, O in lower 27 bits
    out[gid] = (x << 27) | o;
}
