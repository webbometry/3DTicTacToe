package precomputing.minimax.canonicalization;

import java.util.AbstractMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentMap;
import java.util.stream.Collectors;

public class PermutationCanonicalizer {

    /**
     * Collapse any boards that are pure permutations of one another.
     * Keeps the first one encountered (relative to the parallel stream).
     */
    public List<String> filter(List<String> boards) {
        ConcurrentMap<String, String> sigToBoard = boards.parallelStream()
                .map(b -> {
                    char[] arr = b.toCharArray();
                    java.util.Arrays.sort(arr);
                    return new AbstractMap.SimpleEntry<>(new String(arr), b);
                })
                .collect(Collectors.toConcurrentMap(
                        Map.Entry::getKey,
                        Map.Entry::getValue,
                        (existing, replacement) -> existing
                ));

        return List.copyOf(sigToBoard.values());
    }
}