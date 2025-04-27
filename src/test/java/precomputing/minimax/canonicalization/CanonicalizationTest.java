package precomputing.minimax.canonicalization;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import precomputing.minimax.kernels.Canonicalization;

import java.nio.file.Paths;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.*;

class CanonicalizationTest {

    private static Canonicalization pipeline;

    @BeforeAll
    static void setup() throws Exception {
        pipeline = new Canonicalization(Paths.get("src/main/data/rotationMaps.txt"));
    }

    @Test
    void fullPipelineFiltersAllStages() {
        // Mix of boards:
        //  - duplicates: "AAB", "AAB"
        //  - permutations of ABC: "ABC", "BCA", "CAB"
        //  - permutations of XYZ: "XYZ", "ZYX"
        //  - unique non-equivalent: "MNO"
        List<String> input = List.of("AAB", "AAB", "ABC", "BCA", "CAB", "XYZ", "ZYX", "MNO");
        List<String> out = pipeline.canonicalize(input);

        // Expect one rep for ABC-group, one for XYZ-group, one "MNO"
        assertEquals(3, out.size(), "Should end up with exactly 3 boards");

        // Check no invalid (duplicate‐containing) boards present
        assertTrue(out.stream().noneMatch(s -> {
            // any repeated character → bad
            return s.chars().distinct().count() != s.length();
        }), "Output must have no repeated chars");

        // Check one representative per signature
        Set<String> sigs = out.stream()
                .map(b -> {
                    char[] a = b.toCharArray();
                    java.util.Arrays.sort(a);
                    return new String(a);
                })
                .collect(Collectors.toSet());

        assertEquals(Set.of("ABC", "MNO", "XYZ"), sigs,
                "Signatures must be exactly ABC, MNO, XYZ");
    }
}