#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define MAX_DEPTH   27
#define NUM_UPPER   45
#define NUM_LOWER   45

// Precomputed winning triples for X (uppercase + '.') and O (lowercase + ',')
__constant char winUpper[NUM_UPPER][3] = {
    {'A','B','C'},{'A','D','G'},{'A','E','I'},{'A','J','S'},{'A','K','U'},
    {'A','M','Y'},{'B','E','H'},{'B','K','T'},{'B','N','Z'},{'C','F','I'},
    {'C','L','U'},{'C','O','.'},{'D','E','F'},{'D','M','V'},{'D','N','X'},
    {'E','N','W'},{'F','O','X'},{'G','E','C'},{'G','H','I'},{'G','P','Y'},
    {'G','Q','.'},{'H','Q','Z'},{'I','R','.'},{'J','K','L'},{'J','M','P'},
    {'J','N','R'},{'K','N','Q'},{'L','O','R'},{'M','N','O'},{'P','N','L'},
    {'P','Q','R'},{'S','K','C'},{'S','M','G'},{'S','T','U'},{'S','V','Y'},
    {'S','W','.'},{'T','N','H'},{'T','W','Z'},{'U','O','I'},{'U','X','.'},
    {'V','N','F'},{'V','W','X'},{'Y','Q','I'},{'Y','W','U'},{'Y','Z','.'}
};

__constant char winLower[NUM_LOWER][3] = {
    {'a','b','c'},{'a','d','g'},{'a','e','i'},{'a','j','s'},{'a','k','u'},
    {'a','m','y'},{'b','e','h'},{'b','k','t'},{'b','n','z'},{'c','f','i'},
    {'c','l','u'},{'c','o',','},{'d','e','f'},{'d','m','v'},{'d','n','x'},
    {'e','n','w'},{'f','o','x'},{'g','e','c'},{'g','h','i'},{'g','p','y'},
    {'g','q',','},{'h','q','z'},{'i','r',','},{'j','k','l'},{'j','m','p'},
    {'j','n','r'},{'k','n','q'},{'l','o','r'},{'m','n','o'},{'p','n','l'},
    {'p','q','r'},{'s','k','c'},{'s','m','g'},{'s','t','u'},{'s','v','y'},
    {'s','w',','},{'t','n','h'},{'t','w','z'},{'u','o','i'},{'u','x',','},
    {'v','n','f'},{'v','w','x'},{'y','q','i'},{'y','w','u'},{'y','z',','}
};

/**
 * For each board (zero-padded to MAX_DEPTH), count how many winning lines X has
 * and how many O has.  A terminal board is one where either has >=2 lines, or
 * step == MAX_DEPTH (full board).  Boards before step 9 are never terminal.
 *
 * flags[gid] = 1 if terminal, 0 otherwise.
 */
__kernel void countWins(
    __global const char *inBoards,  // N × MAX_DEPTH
    __global int       *flags,      // N
    const int           step        // current move count
) {
    int gid = get_global_id(0);
    if (step <= 8) {
        flags[gid] = 0;
        return;
    }

    // Load the board moves into local array
    char s[MAX_DEPTH];
    int base = gid * MAX_DEPTH;
    for (int i = 0; i < step; i++) {
        s[i] = inBoards[base + i];
    }

    // Count X lines
    int xLines = 0;
    for (int l = 0; l < NUM_UPPER && xLines < 2; l++) {
        char a = winUpper[l][0], b = winUpper[l][1], c = winUpper[l][2];
        bool ha=false, hb=false, hc=false;
        for (int i = 0; i < step; i++) {
            char cc = s[i];
            if (cc == a) ha = true;
            else if (cc == b) hb = true;
            else if (cc == c) hc = true;
        }
        if (ha && hb && hc) xLines++;
    }

    // Count O lines only if X didn't already reach 2
    int oLines = 0;
    if (xLines < 2) {
        for (int l = 0; l < NUM_LOWER && oLines < 2; l++) {
            char a = winLower[l][0], b = winLower[l][1], c = winLower[l][2];
            bool ha=false, hb=false, hc=false;
            for (int i = 0; i < step; i++) {
                char cc = s[i];
                if (cc == a) ha = true;
                else if (cc == b) hb = true;
                else if (cc == c) hc = true;
            }
            if (ha && hb && hc) oLines++;
        }
    }

    // Terminal if someone has 2+ lines or the board is full
    if (xLines >= 2 || oLines >= 2 || step == MAX_DEPTH) {
        flags[gid] = 1;
    } else {
        flags[gid] = 0;
    }
}
