// CanonicalKernel.cl
// For each board, this kernel applies 24 precomputed rotation maps and chooses the lexicographically smallest board.
__kernel void canonicalizeBoard(__global const char* boards,
                                __global char* canonicalBoards,
                                __global const int* rotationMaps,  // Flat array: 24 mappings each of 27 ints.
                                const int boardCount) {
    const int boardSize = 27;
    const int numRotations = 24;
    int gid = get_global_id(0);
    if (gid >= boardCount) return;

    const char* board = boards + gid * boardSize;

    // Initialize minBoard to the original board.
    char minBoard[27];
    for (int i = 0; i < boardSize; i++){
        minBoard[i] = board[i];
    }

    // Temporary buffer to hold a rotated board.
    char rotated[27];

    // Loop over each of the 24 rotation mappings.
    for (int rot = 0; rot < numRotations; rot++){
        // Pointer to the current rotation map (an array of 27 ints).
        const int* map = rotationMaps + rot * boardSize;

        // Apply the rotation: for each index i in the original board,
        // place board[i] into rotated at position map[i].
        for (int i = 0; i < boardSize; i++){
            rotated[ map[i] ] = board[i];
        }

        // Lexicographical comparison between rotated and current minBoard.
        int cmp = 0;
        for (int i = 0; i < boardSize; i++){
            if (rotated[i] < minBoard[i]) { cmp = -1; break; }
            else if (rotated[i] > minBoard[i]) { cmp = 1; break; }
        }
        // If rotated board is lexicographically smaller, update minBoard.
        if (cmp < 0) {
            for (int i = 0; i < boardSize; i++){
                minBoard[i] = rotated[i];
            }
        }
    }
    // Write the canonical board to the output buffer.
    for (int i = 0; i < boardSize; i++){
        canonicalBoards[gid * boardSize + i] = minBoard[i];
    }
}
