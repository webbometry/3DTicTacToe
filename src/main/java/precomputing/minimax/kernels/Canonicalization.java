package precomputing.minimax.kernels;

import java.io.IOException;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.ConcurrentMap;
import java.util.stream.Collectors;

/**
 * Fully parallel canonicalization: duplicates → permutations → rotations.
 */
public class Canonicalization {
    private final int[][] rotMaps;      // [27][27]
    private final List<char[]> lowerLines, upperLines;

    public Canonicalization(Path rotationMapsPath) throws IOException {
        // load rotation maps
        var lines = java.nio.file.Files.readAllLines(rotationMapsPath);
        rotMaps = new int[lines.size()][];
        for (int i = 0; i < lines.size(); i++) {
            var parts = lines.get(i).trim().split("\\s+");
            rotMaps[i] = Arrays.stream(parts).mapToInt(Integer::parseInt).toArray();
        }
        // load win-lines if you also want to parallelize CheckWin here...
        lowerLines = upperLines = List.of(); // not used in this class
    }

    /**
     * @param boards raw list of move-sequence strings
     * @return the canonicalized set: no duplicates, no pure permutations,
     * no rotational redundancies.
     */
    public List<String> canonicalize(List<String> boards, int step) {
        // 1) Drop any with repeated chars
        List<String> noDups = boards.parallelStream()
                .filter(this::hasNoRepeats)
                .collect(Collectors.toList());

        // 2) Collapse pure permutations
        ConcurrentMap<String, String> sigToBoard = noDups.parallelStream()
                .map(b -> {
                    char[] a = b.toCharArray();
                    Arrays.sort(a);
                    return new AbstractMap.SimpleEntry<>(new String(a), b);
                })
                .collect(Collectors.toConcurrentMap(
                        Map.Entry::getKey,
                        Map.Entry::getValue,
                        (keep, drop) -> keep
                ));
        List<String> noPerms = List.copyOf(sigToBoard.values());

        // 3) GPU rotation‐canonicalization
        List<String> rotated = new RotationCanonicalizerGPU().canonicalize(noPerms, step);

        // 4) Dedupe canonical results
        return rotated.parallelStream()
                .distinct()
                .collect(Collectors.toList());
    }

    /**
     * Returns true iff no character appears twice in b.
     */
    private boolean hasNoRepeats(String b) {
        boolean[] up = new boolean[26];
        boolean[] lo = new boolean[26];
        boolean dot = false, comma = false;
        for (char c : b.toCharArray()) {
            if (c >= 'A' && c <= 'Z') {
                int i = c - 'A';
                if (up[i]) return false;
                up[i] = true;
            } else if (c >= 'a' && c <= 'z') {
                int i = c - 'a';
                if (lo[i]) return false;
                lo[i] = true;
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
     * Returns the lexicographically smallest rotation of b under your rotMaps.
     */
    private String canonicalRotation(String b) {
        String best = null;
        int L = b.length();
        for (int[] map : rotMaps) {
            char[] tmp = new char[L];
            for (int i = 0; i < L; i++) {
                char c = b.charAt(i), out;
                int idx = (c == '.' || c == ',') ? 26
                        : (c >= 'A' && c <= 'Z') ? (c - 'A')
                        : (c - 'a');
                int r = map[idx];
                if (c >= 'A' && c <= 'Z') {
                    out = (r == 26) ? '.' : (char) ('A' + r);
                } else {
                    out = (r == 26) ? ',' : (char) ('a' + r);
                }
                tmp[i] = out;
            }
            String cand = new String(tmp);
            if (best == null || cand.compareTo(best) < 0) {
                best = cand;
            }
        }
        return best;
    }
}
