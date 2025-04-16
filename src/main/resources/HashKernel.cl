// HashKernel.cl
// Each board is represented as 27 characters.
// ' ' (empty) is encoded as 0, 'x' (or 'X') as 1, and 'o' (or 'O') as 2.
// The resulting 54-bit number is shifted left by 10 bits to reserve room for a score.
__kernel void hashBoard(__global const char* boards,
                        __global ulong* hashes,
                        const int boardCount) {
    const int boardSize = 27;
    int gid = get_global_id(0);
    if (gid >= boardCount) return;

    const char* board = boards + gid * boardSize;
    ulong hash = 0;
    for (int i = 0; i < boardSize; i++) {
        int value = 0;
        char c = board[i];
        if(c == 'x' || c == 'X') {
            value = 1;
        } else if(c == 'o' || c == 'O') {
            value = 2;
        }
        // Shift the accumulated hash left 2 bits and add the new cell value.
        hash = (hash << 2) | value;
    }
    // Shift left by 10 bits to reserve space for the score.
    hash = hash << 10;
    hashes[gid] = hash;
}
