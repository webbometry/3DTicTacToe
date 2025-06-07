#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

// WIN_MASKS[] injected by CLContext

__kernel void expand_and_classify(
    __global const ulong*  inBoards,
    const uint             inCount,
    const uint             depth,

    __global ulong*        outFrontier,
    __global atomic_ulong* frontierIdx,

    __global ulong*        outTermX,
    __global atomic_ulong* termXIdx,

    __global ulong*        outTermO,
    __global atomic_ulong* termOIdx
) {
    uint gid      = get_global_id(0);
    uint bIdx     = gid / 27;
    uint bitIndex = gid % 27;
    if (bIdx >= inCount) return;

    // unpack 54-bit board
    ulong board = inBoards[bIdx];
    ulong xBits  =  board           & 0x7FFFFFFUL;
    ulong oBits  = (board >> 27)    & 0x7FFFFFFUL;

    bool isXturn = (depth & 1) == 0;
    ulong curr   = isXturn ? xBits : oBits;
    ulong other  = isXturn ? oBits   : xBits;
    ulong mask   = 1UL << bitIndex;

    if ((curr & mask) || (other & mask)) return; // occupied

    // flip one bit
    ulong newCurr  = curr | mask;
    ulong newX     = isXturn ? newCurr : xBits;
    ulong newO     = isXturn ? oBits    : newCurr;
    ulong newBoard = newX | (newO << 27);

    // count up to 2 lines
    uint winX = 0, winO = 0;
    uint N     = sizeof(WIN_MASKS)/sizeof(WIN_MASKS[0]);
    for (uint i = 0; i < N; i++) {
        ulong m = WIN_MASKS[i];
        if ((newX & m) == m && ++winX == 2) {
            // X wins: atomically reserve a slot in termXIdx
            ulong idx = atom_inc((__global volatile ulong*)termXIdx);
            outTermX[idx] = newBoard;
            return;
        }
        if ((newO & m) == m && ++winO == 2) {
            // O wins
            ulong idx = atom_inc((__global volatile ulong*)termOIdx);
            outTermO[idx] = newBoard;
            return;
        }
    }

    // not terminal → enqueue to frontier
    ulong fpos = atom_inc((__global volatile ulong*)frontierIdx);
    outFrontier[fpos] = newBoard;
}
