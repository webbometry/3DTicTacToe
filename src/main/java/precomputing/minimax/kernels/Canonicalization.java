package precomputing.minimax.kernels;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.ConcurrentMap;
import java.util.stream.Collectors;

public class Canonicalization {
    private static final int MAX_CELLS = 27;
    private static final int FULL_MASK_BITS = 27;
    private final int[][] rotMaps;  // [24][27]

    public Canonicalization(Path rotationMapsPath) throws IOException {
        List<String> lines = Files.readAllLines(rotationMapsPath);
        rotMaps = new int[lines.size()][];
        for (int i = 0; i < lines.size(); i++) {
            String[] parts = lines.get(i).trim().split("\\s+");
            rotMaps[i] = Arrays.stream(parts).mapToInt(Integer::parseInt).toArray();
        }
    }

    /**
     * @param boards list of move-sequence strings (each char is A–Z, a–z, '.' or ',')
     * @param step   length of each string in boards
     * @return one representative string per equivalence class
     */
    public List<String> canonicalize(List<String> boards, int step) {
        // 1) Drop any with repeated chars (fast parallel scan)
        List<String> noDups = boards.parallelStream()
                .filter(this::hasNoRepeats)
                .collect(Collectors.toList());

        // 2) Collapse pure permutations: build a 54-bit signature, keep first string
        ConcurrentMap<Long, String> permMap = noDups.parallelStream()
                .map(b -> Map.entry(signature(b), b))
                .collect(Collectors.toConcurrentMap(
                        Map.Entry::getKey,
                        Map.Entry::getValue,
                        (keep, drop) -> keep
                ));
        List<String> noPerms = List.copyOf(permMap.values());

        // 3) Collapse rotations: for each board, compute its minimal 54-bit rotated mask
        ConcurrentMap<Long, String> rotMap = noPerms.parallelStream()
                .map(b -> {
                    // build the base bit-masks
                    long baseX = 0, baseO = 0;
                    for (char c : b.toCharArray()) {
                        if (c >= 'A' && c <= 'Z') baseX |= 1L << (c - 'A');
                        else if (c >= 'a' && c <= 'z') baseO |= 1L << (c - 'a');
                        else if (c == '.') baseX |= 1L << 26;
                        else if (c == ',') baseO |= 1L << 26;
                    }

                    // find the lexicographically minimal rotation‐code
                    long best = Long.MAX_VALUE;
                    for (int[] map : rotMaps) {
                        long x = 0, o = 0;

                        // CORRECT: remap every occupied cell
                        long bits = baseX | baseO;
                        while (bits != 0) {
                            int src = Long.numberOfTrailingZeros(bits);
                            bits &= bits - 1;
                            if (((baseX >>> src) & 1) == 1) {
                                x |= 1L << map[src];
                            } else {
                                o |= 1L << map[src];
                            }
                        }

                        long code = (x << FULL_MASK_BITS) | o;
                        if (code < best) best = code;
                    }

                    return Map.entry(best, b);
                })
                .collect(Collectors.toConcurrentMap(
                        Map.Entry::getKey,
                        Map.Entry::getValue,
                        (keep, drop) -> keep
                ));

        // 4) Return the representative strings
        return List.copyOf(rotMap.values());
    }

    /**
     * Returns true iff no character appears twice in the string.
     */
    private boolean hasNoRepeats(String b) {
        boolean[] up = new boolean[26], lo = new boolean[26];
        boolean dot = false, comma = false;
        for (int i = 0; i < b.length(); i++) {
            char c = b.charAt(i);
            if (c >= 'A' && c <= 'Z') {
                int idx = c - 'A';
                if (up[idx]) return false;
                up[idx] = true;
            } else if (c >= 'a' && c <= 'z') {
                int idx = c - 'a';
                if (lo[idx]) return false;
                lo[idx] = true;
            } else if (c == '.') {
                if (dot) return false;
                dot = true;
            } else if (c == ',') {
                if (comma) return false;
                comma = true;
            }
        }
        return true;
    }

    /**
     * Build a 54-bit signature for the multiset of moves in b:
     * upper 27 bits for X, lower 27 for O.
     */
    private long signature(String b) {
        long up = 0, lo = 0;
        for (char c : b.toCharArray()) {
            if (c >= 'A' && c <= 'Z') up |= 1L << (c - 'A');
            else if (c >= 'a' && c <= 'z') lo |= 1L << (c - 'a');
            else if (c == '.') up |= 1L << 26;
            else if (c == ',') lo |= 1L << 26;
        }
        return (up << FULL_MASK_BITS) | lo;
    }
}
