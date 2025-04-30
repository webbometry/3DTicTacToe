// SortAndUniqueBoards.cl
// For each board-string, perform an in-place case-insensitive sort (retaining original case).

__kernel void sortBoard(
    __global const uchar* inBoards,  // [N * maxLen]
    __global       uchar* outBoards, // [N * maxLen]
    const int           maxLen
) {
    int gid = get_global_id(0);
    const uchar* src = inBoards  + gid * maxLen;
          uchar* dst = outBoards + gid * maxLen;

    // load into local temp array
    uchar temp[32];  // assume maxLen ≤ 32 for safety
    int  len = 0;
    for (int i = 0; i < maxLen; i++) {
        uchar c = src[i];
        if (c == 0) break;
        temp[len++] = c;
    }

    // simple insertion sort by lowercase key
    for (int i = 1; i < len; i++) {
        uchar key = temp[i];
        int j = i - 1;
        uchar keyLC = (key >= 'A' && key <= 'Z') ? key + 32 : key;
        while (j >= 0) {
            uchar cj   = temp[j];
            uchar cjLC = (cj  >= 'A' && cj  <= 'Z') ? cj  + 32 : cj;
            if (cjLC <= keyLC) break;
            temp[j+1] = temp[j];
            j--;
        }
        temp[j+1] = key;
    }

    // write back (zero-pad)
    for (int i = 0; i < maxLen; i++) {
        dst[i] = (i < len) ? temp[i] : 0;
    }
}
