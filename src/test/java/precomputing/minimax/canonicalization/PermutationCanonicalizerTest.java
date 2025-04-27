package precomputing.minimax.canonicalization;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.*;

class PermutationCanonicalizerTest {

    private final PermutationCanonicalizer canon = new PermutationCanonicalizer();

    @Test
    void collapsesPurePermutations() {
        List<String> input = List.of("ABC", "BCA", "CAB", "ACB", "XYZ", "YZX");
        List<String> out = canon.filter(input);

        // Should end up with exactly two signatures: "ABC" and "XYZ"
        assertEquals(2, out.size(), "Should collapse the 3 ABC perms into 1, and 2 XYZ perms into 1");

        // Check that each retained board’s sorted signature is either "ABC" or "XYZ"
        List<String> sigs = out.stream()
                .map(b -> {
                    char[] arr = b.toCharArray();
                    java.util.Arrays.sort(arr);
                    return new String(arr);
                })
                .collect(Collectors.toList());

        assertTrue(sigs.contains("ABC"), "One representative must have signature 'ABC'");
        assertTrue(sigs.contains("XYZ"), "One representative must have signature 'XYZ'");
    }

    @Test
    void keepsDistinctSignatures() {
        List<String> input = List.of("AAB", "ABA", "ABC", "CAB", "XYY", "YYX");
        List<String> out = canon.filter(input);

        // Signatures are "AAB", "ABC", "XYY" => 3 groups
        assertEquals(3, out.size(), "Should yield one per distinct signature");

        List<String> sigs = out.stream()
                .map(b -> {
                    char[] arr = b.toCharArray();
                    java.util.Arrays.sort(arr);
                    return new String(arr);
                })
                .collect(Collectors.toList());

        assertTrue(sigs.contains("AAB"));
        assertTrue(sigs.contains("ABC"));
        assertTrue(sigs.contains("XYY"));
    }
}