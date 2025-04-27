package precomputing.minimax.canonicalization;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.AbstractMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentMap;
import java.util.stream.Collectors;

public class RotationCanonicalizer {
    private final int[][] rotMaps;  // [27][27]

    public RotationCanonicalizer(Path rotationMapsPath) throws IOException {
        var lines = Files.readAllLines(rotationMapsPath);
        rotMaps = new int[lines.size()][];
        for (int i = 0; i < lines.size(); i++) {
            var parts = lines.get(i).trim().split("\\s+");
            rotMaps[i] = new int[parts.length];
            for (int j = 0; j < parts.length; j++) {
                rotMaps[i][j] = Integer.parseInt(parts[j]);
            }
        }
    }

    /**
     * Collapse rotationally equivalent boards.
     * Keeps the first board encountered for each canonical rotation.
     */
    public List<String> canonicalize(List<String> boards) {
        ConcurrentMap<String, String> canonToBoard = boards.parallelStream()
                .map(b -> new AbstractMap.SimpleEntry<>(canonicalRotation(b), b))
                .collect(Collectors.toConcurrentMap(
                        Map.Entry::getKey,
                        Map.Entry::getValue,
                        (existing, replacement) -> existing
                ));

        return List.copyOf(canonToBoard.values());
    }

    /**
     * Compute the lexicographically minimal rotation of b.
     */
    private String canonicalRotation(String b) {
        String best = null;
        int L = b.length();
        for (int[] map : rotMaps) {
            char[] tmp = new char[L];
            for (int i = 0; i < L; i++) {
                char c = b.charAt(i);
                int idx;
                if (c >= 'A' && c <= 'Z') idx = c - 'A';
                else if (c >= 'a' && c <= 'z') idx = c - 'a';
                else if (c == '.' || c == ',') idx = 26;
                else throw new IllegalArgumentException("Invalid char: " + c);

                int r = map[idx];
                if (c >= 'A' && c <= 'Z') {
                    tmp[i] = (r == 26 ? '.' : (char) ('A' + r));
                } else {
                    tmp[i] = (r == 26 ? ',' : (char) ('a' + r));
                }
            }
            String cand = new String(tmp);
            if (best == null || cand.compareTo(best) < 0) {
                best = cand;
            }
        }
        return best;
    }
}