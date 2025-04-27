package precomputing.minimax.kernels;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.*;

class ExpansionTest {
    private Expansion expansion;

    @BeforeEach
    void setUp() {
        expansion = new Expansion();
    }

    /**
     * 1) Expansion simply multiplies the input count by 27.
     */
    @Test
    void testProduces27PerInput() {
        List<String> input = List.of("", "A", "BC");
        List<String> out = expansion.expandAll(input);
        assertEquals(
                input.size() * 27,
                out.size(),
                "Should produce exactly 27 children per input board"
        );
    }

    /**
     * 2) On X's turn (even-length prefix), appended char ∈ {'A'..'Z','.'}.
     */
    @Test
    void testXTurnAppendedChars() {
        String prefix = "";        // length=0 → X's turn
        List<String> out = expansion.expandAll(List.of(prefix));

        // collect the last char of each child
        Set<Character> appended = out.stream()
                .map(s -> s.charAt(s.length() - 1))
                .collect(Collectors.toSet());

        // expected = 'A'..'Z' plus '.'
        Set<Character> expected = IntStream.rangeClosed('A', 'Z')
                .mapToObj(c -> (char) c)
                .collect(Collectors.toSet());
        expected.add('.');

        assertEquals(expected, appended,
                "X-turn must append exactly A–Z and '.'");
    }

    /**
     * 3) On O's turn (odd-length prefix), appended char ∈ {'a'..'z',','}.
     */
    @Test
    void testOTurnAppendedChars() {
        String prefix = "A";      // length=1 → O's turn
        List<String> out = expansion.expandAll(List.of(prefix));

        Set<Character> appended = out.stream()
                .map(s -> s.charAt(s.length() - 1))
                .collect(Collectors.toSet());

        Set<Character> expected = IntStream.rangeClosed('a', 'z')
                .mapToObj(c -> (char) c)
                .collect(Collectors.toSet());
        expected.add(',');

        assertEquals(expected, appended,
                "O-turn must append exactly a–z and ','");
    }

    /**
     * 4) Every child string = parent prefix + exactly one new character.
     */
    @Test
    void testPrefixPreservedAndLength() {
        String prefix = "XYZ";
        List<String> out = expansion.expandAll(List.of(prefix));

        assertEquals(27, out.size());
        for (String child : out) {
            assertTrue(
                    child.startsWith(prefix),
                    () -> "Child '" + child + "' must start with prefix '" + prefix + "'"
            );
            assertEquals(
                    prefix.length() + 1,
                    child.length(),
                    () -> "Child '" + child + "' must have length " + (prefix.length() + 1)
            );
        }
    }

    /**
     * 5) Outputs are grouped in the same order as the inputs:
     * first 27 for input[0], next 27 for input[1], etc.
     */
    @Test
    void testGroupingOrder() {
        List<String> input = List.of("", "A", "BC");
        List<String> out = expansion.expandAll(input);
        assertEquals(3 * 27, out.size());

        // first group → X-turn children of ""
        List<String> g0 = out.subList(0, 27);
        assertTrue(g0.stream().allMatch(s -> s.length() == 1),
                "First group children all length=1");

        // second group → children of "A", should all start with "A"
        List<String> g1 = out.subList(27, 54);
        assertTrue(g1.stream().allMatch(s -> s.startsWith("A")),
                "Second group must all start with \"A\"");

        // third group → children of "BC", should all start with "BC"
        List<String> g2 = out.subList(54, 81);
        assertTrue(g2.stream().allMatch(s -> s.startsWith("BC")),
                "Third group must all start with \"BC\"");
    }

    /**
     * 6) Determinism: repeated calls on the same input yield identical lists.
     */
    @Test
    void testDeterministic() {
        List<String> input = List.of("", "A", "HELLO");
        List<String> first = expansion.expandAll(input);
        List<String> second = expansion.expandAll(input);
        assertEquals(first, second, "Expansion must be purely deterministic");
    }
}