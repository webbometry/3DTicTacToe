#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

__kernel void expandBoards(
    __global const char* boards,    // input: N × 27 chars
    __global char*       output,    // output: up to N*27 × 27 chars
    __global int*        counter,   // atomic counter
    const uchar          player,    // 'x' or 'o'
    const int            boardCount
) {
    int gid = get_global_id(0);
    if (gid >= boardCount) return;

    const int BOARD_SIZE = 27;
    __global const char* inB = boards + gid * BOARD_SIZE;

    // For each empty cell, attempt a move
    for (int idx = 0; idx < BOARD_SIZE; idx++) {
        if (inB[idx] == ' ') {
            // Reserve an index via atomic_inc on our plain int*
            int outIdx = atomic_inc(counter);
            __global char* outB = output + outIdx * BOARD_SIZE;

            // Copy and place the move
            for (int i = 0; i < BOARD_SIZE; i++) {
                outB[i] = inB[i];
            }
            outB[idx] = player;
        }
    }
}