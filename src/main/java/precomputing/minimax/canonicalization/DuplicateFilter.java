package precomputing.minimax.canonicalization;

import java.util.List;
import java.util.stream.Collectors;

public class DuplicateFilter {

    /**
     * Remove any board containing a repeated character (case-sensitive).
     */
    public List<String> filter(List<String> boards) {
        return boards.parallelStream()
                .filter(b -> {
                    boolean[] up = new boolean[26];
                    boolean[] lo = new boolean[26];
                    boolean dot = false;
                    boolean comma = false;
                    for (int i = 0, len = b.length(); i < len; i++) {
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
                })
                .collect(Collectors.toList());
    }
}