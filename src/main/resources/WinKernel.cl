// WinKernel.cl

// Each bitboard is 54 bits: lower 27 bits = X, next 27 bits = O, then shifted left by SCORE_BITS.
// We'll ignore the lower SCORE_BITS bits.
__kernel void checkWin(
    __global const ulong* hashes,
    __global const int* winLines,   // length = numLines * 3
    const int numLines,
    __global uchar* gameOver,       // output flags
    __global uchar* winner          // 0=none/draw, 1=X, 2=O
) {
    int gid = get_global_id(0);

    // Extract the board bits (ignore score bits).
    // state = bits[63:0], actual board bits = state >> SCORE_BITS
    ulong state = hashes[gid] >> 10;

    // Split into X and O bitboards.
    uint xBits = (uint)(state & ((1UL<<27)-1));
    uint oBits = (uint)((state >> 27) & ((1UL<<27)-1));

    int xCount = 0;
    int oCount = 0;

    // For each win-line, test the three positions.
    for (int i = 0; i < numLines; i++) {
        int base = 3*i;
        int b0 = winLines[base + 0];
        int b1 = winLines[base + 1];
        int b2 = winLines[base + 2];
        // mask for these three bits
        uint mask = (1u << b0) | (1u << b1) | (1u << b2);
        // if all three in X:
        if ((xBits & mask) == mask) {
            xCount++;
            if (xCount >= 2) {
                gameOver[gid] = 1;
                winner[gid] = 1;
                return;
            }
        }
        if ((oBits & mask) == mask) {
            oCount++;
            if (oCount >= 2) {
                gameOver[gid] = 1;
                winner[gid] = 2;
                return;
            }
        }
    }

    // No two‐line win found: check for draw (board full)
    // board full if xCount+oCount == 27
    // but we don't know count of bits; instead test (xBits | oBits) == all 27 bits set
    if ((xBits | oBits) == ((1u<<27)-1)) {
        gameOver[gid] = 1;
        winner[gid]   = 0;  // draw
    } else {
        gameOver[gid] = 0;
        winner[gid]   = 0;
    }
}
