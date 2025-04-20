// CheckWinKernel.cl
//
// This kernel reads each partial move‐sequence (inSeqs), reconstructs the
// 3×3×3 board, and then checks for a “two‐line” win by either X or O.
// Win‐lines are stored as character triples in two buffers:
//   winUpper — uppercase 'A'..'Z','.' triples for X‐lines
//   winLower — lowercase 'a'..'z',',' triples for O‐lines
//
// Arguments:
//   inSeqs       - __global const char* : N sequences, each of length 'step'
//   winUpper     - __global const char* : W×3 chars ('A'..'Z','.')
//   winLower     - __global const char* : W×3 chars ('a'..'z',',')
//   const int W  - number of win‐lines (each is 3 chars in the above buffers)
//   const int N  - number of sequences (boards) to check
//   const int step - length of each sequence (number of moves so far)
//   statusOut    - __global int*       : N status codes (0=playable, 1=win, 2=draw)

__kernel void checkWin(
    __global const char* inSeqs,
    __global const char* winUpper,
    __global const char* winLower,
    const int            W,
    const int            N,
    const int            step,
    __global int*        statusOut)
{
    int gid = get_global_id(0);
    if (gid >= N) return;

    // 1) Reconstruct the board: initialize to empty
    char board[27];
    #pragma unroll
    for (int i = 0; i < 27; i++) {
        board[i] = ' ';
    }

    // 2) Play each move in the sequence
    const char* seq = inSeqs + gid * step;
    for (int m = 0; m < step; m++) {
        char c = seq[m];
        int pos;
        // map symbol → 0..26
        if (c == '.'  || c == ',') {
            pos = 26;
        }
        else if (c >= 'A' && c <= 'Z') {
            pos = c - 'A';
        }
        else { // 'a'..'z'
            pos = c - 'a';
        }
        // set board cell to 'X' or 'O'
        if (c == '.' || (c >= 'A' && c <= 'Z')) {
            board[pos] = 'X';
        } else {
            board[pos] = 'O';
        }
    }

    // 3) Count how many winning lines each player has
    int countX = 0;
    int countO = 0;

    // loop over each precomputed win‑line
    for (int l = 0; l < W; l++) {
        // --- check X‐line ---
        char u0 = winUpper[3*l + 0];
        char u1 = winUpper[3*l + 1];
        char u2 = winUpper[3*l + 2];
        // map back to board indices
        int a = (u0 == '.') ? 26 : (u0 - 'A');
        int b = (u1 == '.') ? 26 : (u1 - 'A');
        int c = (u2 == '.') ? 26 : (u2 - 'A');
        // if all three are 'X', increment
        if (board[a] == 'X' && board[b] == 'X' && board[c] == 'X') {
            countX++;
        }

        // --- check O‐line ---
        char l0 = winLower[3*l + 0];
        char l1 = winLower[3*l + 1];
        char l2 = winLower[3*l + 2];
        int oa = (l0 == ',') ? 26 : (l0 - 'a');
        int ob = (l1 == ',') ? 26 : (l1 - 'a');
        int oc = (l2 == ',') ? 26 : (l2 - 'a');
        if (board[oa] == 'O' && board[ob] == 'O' && board[oc] == 'O') {
            countO++;
        }

        // stop early if either has two lines
        if (countX >= 2 || countO >= 2) {
            break;
        }
    }

    // 4) Decide terminal status
    if (countX >= 2 || countO >= 2) {
        statusOut[gid] = 1;    // win for whichever side
    }
    else if (step == 27) {
        statusOut[gid] = 2;    // full board, draw
    }
    else {
        statusOut[gid] = 0;    // still playable
    }
}
