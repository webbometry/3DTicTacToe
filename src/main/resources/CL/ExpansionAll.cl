#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define MAX_DEPTH 27
#define CHILD_LEN (MAX_DEPTH+1)

// For each board[gid], append all 27 possible moves (A–Z+'.' or a–z+',')
// and write to outBoards at gid*27…(gid*27+26)
__kernel void expandAll(
    __global const char *inBoards,    // N × MAX_DEPTH
    __global const uchar *turnFlags,  // N
    __global char *outBoards          // N×27 × CHILD_LEN
) {
    int gid = get_global_id(0);
    int baseIn  = gid * MAX_DEPTH;
    int flag    = turnFlags[gid];         // 0 = X, 1 = O
    char offset = flag == 0 ? 'A' : 'a';
    char special= flag == 0 ? '.' : ',';

    // compute prefix length
    int len = 0;
    for (; len < MAX_DEPTH; len++) {
        char c = inBoards[baseIn + len];
        if (c == 0) break;
    }

    // for each of 27 moves
    for (int m = 0; m < 27; m++) {
        int baseOut = (gid * 27 + m) * CHILD_LEN;
        // copy prefix
        for (int i = 0; i < len; i++) {
            outBoards[baseOut + i] = inBoards[baseIn + i];
        }
        // append move
        outBoards[baseOut + len] =
            (m == 26 ? special : (char)(offset + m));
        // zero-pad rest
        for (int i = len + 1; i < CHILD_LEN; i++) {
            outBoards[baseOut + i] = 0;
        }
    }
}
