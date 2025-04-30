// FilterInvalidBoards.cl
// Marks boards with duplicate positions (ignoring case, and treating '.'/',' as the same) as invalid.

__kernel void filterValid(
    __global const uchar* inBoards, // [N * maxLen] packed ASCII, zero-padded
    __global       int*   flags,    // [N] 1 = valid, 0 = invalid
    const int           maxLen
) {
    int gid = get_global_id(0);
    const uchar* board = inBoards + gid * maxLen;

    // track seen positions: 0–26
    uchar seen[27] = {0};

    for (int i = 0; i < maxLen; i++) {
        uchar c = board[i];
        if (c == 0) break;               // end of string
        int idx;
        if (c == '.' || c == ',') {
            idx = 26;
        } else {
            // lowercase
            uchar lc = (c >= 'A' && c <= 'Z') ? c + 32 : c;
            idx = lc - 'a';
        }
        if (idx < 0 || idx > 26) continue;  // skip any unexpected chars
        if (seen[idx]) {
            flags[gid] = 0;
            return;
        }
        seen[idx] = 1;
    }
    flags[gid] = 1;
}
