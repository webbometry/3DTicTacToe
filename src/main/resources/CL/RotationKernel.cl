// RotationKernel.cl

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

// hard-code map dimensions
#define MAP_SIZE 27
#define MAP_COUNT 24

__kernel void collapseRotation(
    __global const ulong* inCodes,     // [N] original 54-bit codes
    __global const int*   rotMaps,     // [MAP_COUNT * MAP_SIZE] flattened
    __global       ulong* outMinCodes  // [N] minimal codes
) {
    int gid = get_global_id(0);

    // load base code
    ulong base = inCodes[gid];
    // split into X-mask and O-mask
    ulong baseX = base >> MAP_SIZE;
    ulong baseO = base & ((1UL << MAP_SIZE) - 1);
    // bits set in either mask
    ulong bitsAll = baseX | baseO;

    ulong best = (ulong)(-1); // all-ones = max

    // for each rotation
    for (int r = 0; r < MAP_COUNT; r++) {
        ulong x = 0, o = 0;
        ulong bits = bitsAll;

        // spin through all set bits
        while (bits != 0) {
            // find lowest set bit index
            int src = 0;
            // MAP_SIZE is small (27), so linear scan is fine
            for (; src < MAP_SIZE; src++) {
                if ((bits >> src) & 1UL) break;
            }
            bits &= bits - 1;  // clear that bit

            // where does src go under rotation r?
            int dst = rotMaps[r * MAP_SIZE + src];

            // set in X or O mask
            if ((baseX >> src) & 1UL) {
                x |= (1UL << dst);
            } else {
                o |= (1UL << dst);
            }
        }

        // recombine and keep the smallest
        ulong code = (x << MAP_SIZE) | o;
        if (code < best) best = code;
    }

    outMinCodes[gid] = best;
}
