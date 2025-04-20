// ExpansionKernel.cl
//
// For each partial move‑sequence of length prevLen in 'inMoves',
// appends exactly one legal next move (A..Z/. for X, a..z/, for O),
// writing all length=(prevLen+1) sequences into 'outMoves'.
// 'outCount' is an atomic counter tracking how many new sequences were emitted.

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

__kernel void expandBoards(
    __global const char* inMoves,   // boardCount × prevLen
    __global       char* outMoves,  // maxOut  × (prevLen+1)
    __global       int*  outCount,  // atomic counter
    const int prevLen,              // length of each input sequence
    const uchar nextPlayer,         // 0 ⇒ X’s turn, 1 ⇒ O’s turn
    const int boardCount)           // number of sequences in inMoves
{
    // 27 move symbols for X and O:
    __private const char symbolsX[27] = {
        'A','B','C','D','E','F','G','H','I',
        'J','K','L','M','N','O','P','Q','R',
        'S','T','U','V','W','X','Y','Z','.'
    };
    __private const char symbolsO[27] = {
        'a','b','c','d','e','f','g','h','i',
        'j','k','l','m','n','o','p','q','r',
        's','t','u','v','w','x','y','z',','
    };

    int gid = get_global_id(0);
    if (gid >= boardCount) return;

    // pointer to the start of this board's move‑string
    __global const char* seq = inMoves + gid * prevLen;

    // pick which symbol set to use
    __private const char* symbols = (nextPlayer == 0) ? symbolsX : symbolsO;

    // try each of the 27 positions
    for (int i = 0; i < 27; i++) {
        char c = symbols[i];

        // check if 'c' already appears in seq[0..prevLen-1]
        bool used = false;
        for (int j = 0; j < prevLen; j++) {
            if (seq[j] == c) {
                used = true;
                break;
            }
        }
        if (used) continue;

        // reserve a slot via atomic increment
        int idx = atomic_inc(outCount);

        // write into outMoves[idx], which has length prevLen+1
        __global char* dest = outMoves + idx * (prevLen + 1);

        // copy the existing sequence
        for (int j = 0; j < prevLen; j++) {
            dest[j] = seq[j];
        }
        // append the new symbol
        dest[prevLen] = c;
    }
}
