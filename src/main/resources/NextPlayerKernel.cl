// NextPlayerKernel.cl

__kernel void nextPlayer(
    __global const ulong* hashes,
    const int count,
    __global uchar* nextPlayer  // 1 = X, 2 = O, 0 illegal
) {
    int gid = get_global_id(0);
    if (gid >= count) return;

    // Extract board bits
    ulong state = hashes[gid] >> 10;
    uint xBits = (uint)(state & ((1UL<<27)-1));
    uint oBits = (uint)((state >> 27) & ((1UL<<27)-1));

    // Use OpenCL builtin popcount
    int cx = popcount(xBits);
    int co = popcount(oBits);

    // Whose turn?
    if (cx == co)      nextPlayer[gid] = 'x';
    else if (cx == co+1) nextPlayer[gid] = 'o';
    else                nextPlayer[gid] = 0;  // illegal
}
