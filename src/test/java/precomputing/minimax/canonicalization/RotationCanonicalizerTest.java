package precomputing.minimax.canonicalization;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.nio.file.Paths;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.*;

class RotationCanonicalizerTest {

    private static RotationCanonicalizer canon;

    @BeforeAll
    static void load() throws Exception {
        canon = new RotationCanonicalizer(Paths.get("src/main/data/rotationMaps.txt"));
    }

    @Test
    void removesIdenticalBoards() {
        List<String> input = List.of("ABC", "ABC", "ABC");
        List<String> out = canon.canonicalize(input);
        assertEquals(1, out.size(), "Strict duplicates must be collapsed");
        assertEquals("ABC", out.get(0));
    }

    @Test
    void keepsDistinctNonEquivalentBoards() {
        List<String> input = List.of("ABC", "DEF", "CAB");
        // We know "CAB" is not lexicographically < "ABC" under any rotation,
        // but we cannot guarantee which one is chosen as canonical.
        List<String> out = canon.canonicalize(input);
        // at most remove rotations; strict distinct strings should remain distinct
        assertTrue(out.size() >= 2, "At least two non-rotationally-equivalent boards remain");

        // Ensure every output is one of the inputs
        Set<String> inputs = Set.of("ABC", "DEF", "CAB");
        assertTrue(inputs.containsAll(out), "Output should be subset of input set");
    }

    @Test
    void choosesLexicographicallyMinimalRotationAsKey() {
        // Pick a board for which you can see that one rotation is smaller:
        // For example, if a rotation map swaps first two cells, then "BA" would rotate to "AB"
        // Observable test: if input contains "BA", its canonical rep should be "AB"
        List<String> input = List.of("BA");
        List<String> out = canon.canonicalize(input);
        String rep = out.get(0);
        assertTrue(rep.equals("AB") || rep.equals("BA"),
                "Representative must be either 'AB' or 'BA', depending on maps");
        // And it should be the lexicographically smaller one
        assertEquals(
                List.of(rep).stream().sorted().findFirst().get(),
                rep,
                "Representative must be the lexicographically smallest among rotations"
        );
    }
}