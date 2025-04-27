package precomputing.minimax.canonicalization;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class DuplicateFilterTest {

    private final DuplicateFilter filter = new DuplicateFilter();

    @Test
    void removesUppercaseDuplicates() {
        List<String> input = List.of("AAB", "ABC", "XYZ", "AABC");
        List<String> out = filter.filter(input);
        assertFalse(out.contains("AAB"), "Should remove 'AAB' (two 'A's)");
        assertFalse(out.contains("AABC"), "Should remove 'AABC' (two 'A's)");
        assertTrue(out.contains("ABC"), "Should keep 'ABC'");
        assertTrue(out.contains("XYZ"), "Should keep 'XYZ'");
    }

    @Test
    void removesLowercaseDuplicates() {
        List<String> input = List.of("abc", "aabc", "bcd", "bbc");
        List<String> out = filter.filter(input);
        assertFalse(out.contains("aabc"), "Should remove 'aabc' (two 'a's)");
        assertFalse(out.contains("bbc"), "Should remove 'bbc'  (two 'b's)");
        assertTrue(out.contains("abc"), "Should keep 'abc'");
        assertTrue(out.contains("bcd"), "Should keep 'bcd'");
    }

    @Test
    void removesDotAndCommaDuplicates() {
        List<String> input = List.of(".", ",", "..", ",,", ".,", ",.");
        List<String> out = filter.filter(input);
        assertFalse(out.contains(".."), "Should remove '..'");
        assertFalse(out.contains(",,"),
                "Should remove ',,'");
        // single-character boards are fine
        assertTrue(out.contains("."), "Should keep '.'");
        assertTrue(out.contains(","), "Should keep ','");
        // mixed single occurrences
        assertTrue(out.contains(".,"),
                "Should keep '.,' (one dot & one comma)");
        assertTrue(out.contains(",."),
                "Should keep ',.' (one comma & one dot)");
    }

    @Test
    void keepsBoardsWithoutDuplicates() {
        List<String> input = List.of("ABC", "xyz", "A,z", "MnO");
        List<String> out = filter.filter(input);
        assertEquals(input.size(), out.size());
        assertTrue(out.containsAll(input));
    }
}