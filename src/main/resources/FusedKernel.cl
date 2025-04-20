// src/main/resources/FusedKernel.cl

// rotationMaps: 24 × 27 ints
// winUpper, winLower: W × 3 chars
// boards: N × step chars

__kernel void fuseCanonAndWin(
    __global const char* boards,
    __global uchar*   statuses,      // output[N]: 0=drop,1=survive,2=win,3=draw
    const int N,
    const int step,
    __global const int* rotMaps,
    const int rotMapCount,           // == 24*27
    __global const char* winUpper,
    __global const char* winLower,
    const int W                       // number of win‐lines
) {
    int gid = get_global_id(0);
    if (gid >= N) return;

    // 1) Canonical check:
    //    Build the minimal rotation of boards[gid*step…] across 24 maps.
    __global const char* base = boards + gid*step;
    char best[64];              // step ≤ 27
    for (int i = 0; i < step; i++) best[i] = base[i];

    for (int r = 1; r < 24; r++) {
        __global const int* map = rotMaps + r*27;
        char tmp[64];
        for (int i = 0; i < step; i++) {
            // map[i] in [0..26]
            int idx = map[i];
            if (idx < step) tmp[i] = base[idx];
            else            tmp[i] = ' '; // should never happen
        }
        // lexicographically compare tmp vs best
        bool smaller = false;
        for (int i = 0; i < step; i++) {
            if (tmp[i] < best[i]) { smaller = true; break; }
            if (tmp[i] > best[i]) break;
        }
        if (smaller) {
            for (int i = 0; i < step; i++) best[i] = tmp[i];
        }
    }
    // If this rotation is not equal to the original, drop.
    bool isCanon = true;
    for (int i = 0; i < step; i++) {
        if (best[i] != base[i]) { isCanon = false; break; }
    }
    if (!isCanon) {
        statuses[gid] = 0;
        return;
    }

    // 2) Win‑check on the canonical board:
    //    Count how many win‐lines satisfied.
    int lines = 0;
    for (int w = 0; w < W; w++) {
        __global const char* u = winUpper + w*3;
        __global const char* l = winLower + w*3;
        // check upper pattern
        bool matchU = true;
        for (int k = 0; k < 3; k++) {
            char c = u[k];
            bool ok = false;
            for (int i = 0; i < step; i++) {
                if (base[i] == c) { ok = true; break; }
            }
            if (!ok) { matchU = false; break; }
        }
        if (matchU) { lines++; if (lines >= 2) break; }
        // check lower pattern
        bool matchL = true;
        for (int k = 0; k < 3; k++) {
            char c = l[k];
            bool ok = false;
            for (int i = 0; i < step; i++) {
                if (base[i] == c) { ok = true; break; }
            }
            if (!ok) { matchL = false; break; }
        }
        if (matchL) { lines++; if (lines >= 2) break; }
    }

    // 3) Status:
    //    step == MAX (27) && lines < 2 ⇒ draw
    if (lines >= 2) {
        statuses[gid] = 2;  // win
    } else if (step == 27) {
        statuses[gid] = 3;  // draw
    } else {
        statuses[gid] = 1;  // survive
    }
}
