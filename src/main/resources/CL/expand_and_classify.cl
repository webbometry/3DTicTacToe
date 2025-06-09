#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

// WIN_MASKS[] injected by CLContext

__kernel void expand_and_classify(
    __global const ulong*  inBoards,
    const uint             inCount,
    const uint             depth,

    // frontier
    __global ulong*        outFrontier,
    __global atomic_ulong* frontierIdx,

    // X-win
    __global ulong*        outTermX,
    __global atomic_ulong* termXIdx,

    // O-win
    __global ulong*        outTermO,
    __global atomic_ulong* termOIdx,

    // Tie (full board)
    __global ulong*        outTermTie,
    __global atomic_ulong* termTieIdx
) {
    uint gid      = get_global_id(0);
    uint bIdx     = gid / 27;
    uint bitIndex = gid % 27;
    if (bIdx >= inCount) return;

    // unpack
    ulong board = inBoards[bIdx];
    ulong xBits  =  board           & 0x7FFFFFFUL;
    ulong oBits  = (board >> 27)    & 0x7FFFFFFUL;

    bool isXturn = (depth & 1) == 0;
    ulong curr   = isXturn ? xBits : oBits;
    ulong other  = isXturn ? oBits   : xBits;
    ulong mask   = 1UL << bitIndex;

    // occupied?
    if ((curr & mask) || (other & mask)) return;

    // flip it
    ulong newCurr  = curr | mask;
    ulong newX     = isXturn ? newCurr : xBits;
    ulong newO     = isXturn ? oBits    : newCurr;
    ulong newBoard = newX | (newO << 27);

    // 1) check two 3-in-a-rows → X-win or O-win
    uint winX = 0, winO = 0;
    uint WN = sizeof(WIN_MASKS)/sizeof(WIN_MASKS[0]);
    for (uint i = 0; i < WN; i++) {
        ulong m = WIN_MASKS[i];
        if ((newX & m) == m && ++winX == 2) {
            ulong idx = atom_inc((__global volatile ulong*)termXIdx);
            outTermX[idx] = newBoard;
            return;
        }
        if ((newO & m) == m && ++winO == 2) {
            ulong idx = atom_inc((__global volatile ulong*)termOIdx);
            outTermO[idx] = newBoard;
            return;
        }
    }

    // 2) draw if full (27 bits set)
    if (((newX | newO) == 0x7FFFFFFUL)) {
        ulong idx = atom_inc((__global volatile ulong*)termTieIdx);
        outTermTie[idx] = newBoard;
        return;
    }

    // 3) otherwise frontier
    ulong fpos = atom_inc((__global volatile ulong*)frontierIdx);
    outFrontier[fpos] = newBoard;
}
