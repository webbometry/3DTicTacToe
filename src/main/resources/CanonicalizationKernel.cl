// drop any move‐sequence whose board‑state is not lexicographically minimal
// under the 24 rotations in rotationMaps
__kernel void canonicalize(
    __global const char* inSeqs,      // [N × step]
    __global const int*  rotationMaps,// [24 × 27]
    const int            N,
    const int            step,
    __global uchar*      isCanon)     // out flags [N]
{
    int gid = get_global_id(0);
    const char* seq = inSeqs + gid * step;

    // build original as one string
    // lexCompare with each rotated variant
    bool keep = true;

    // buffer for rotated
    char rot[64]; // step ≤ 27

    for (int r = 0; r < 24 && keep; r++) {
        // build rotated sequence
        for (int m = 0; m < step; m++) {
            char c = seq[m];
            int ord, idx;
            bool isX = (c == '.' || (c >= 'A' && c <= 'Z'));
            if (c == '.' || c == ',') {
                ord = 26;
            } else {
                ord = isX ? (c - 'A') : (c - 'a');
            }
            idx = rotationMaps[r * 27 + ord];
            // map back to symbol
            if (isX) {
                rot[m] = (idx == 26 ? '.' : (char)('A' + idx));
            } else {
                rot[m] = (idx == 26 ? ',' : (char)('a' + idx));
            }
        }
        // lex compare: if rot < seq → drop
        for (int m = 0; m < step; m++) {
            if (rot[m] < seq[m]) { keep = false; break; }
            if (rot[m] > seq[m]) break;
        }
    }
    isCanon[gid] = keep ? 1 : 0;
}
