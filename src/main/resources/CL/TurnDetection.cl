#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define MAX_DEPTH 27

// For each board, set turnFlags[gid] = 0 if X to move, 1 if O to move.
__kernel void detectTurn(
    __global const char *inBoards,   // N × MAX_DEPTH, zero-padded
    __global uchar *turnFlags        // N flags
) {
    int gid = get_global_id(0);
    int base = gid * MAX_DEPTH;
    // find length up to first zero
    int len = 0;
    for (; len < MAX_DEPTH; len++) {
        if (inBoards[base + len] == 0) break;
    }
    // even length → X’s move (flag=0), odd → O’s move (flag=1)
    turnFlags[gid] = (uchar)(len & 1);
}
