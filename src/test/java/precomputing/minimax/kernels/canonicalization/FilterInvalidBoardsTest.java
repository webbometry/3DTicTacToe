// src/test/java/precomputing/minimax/kernels/canonicalization/FilterInvalidBoardsTest.java
package precomputing.minimax.kernels.canonicalization;

import org.jocl.CL;
import org.junit.jupiter.api.*;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class FilterInvalidBoardsTest {
    private static FilterInvalidBoards filter;

    @BeforeAll
    static void setUp() {
        CL.setExceptionsEnabled(true);
        filter = new FilterInvalidBoards();
    }

    @AfterAll
    static void tearDown() {
        filter.release();
    }

    @Test
    void testEmpty() {
        assertTrue(filter.filter(List.of()).isEmpty(),
                "Empty input → empty output");
    }

    @Test
    void testAllValidSingleChars() {
        var input = List.of("abc", "XYZ", ".");
        var out = filter.filter(input);
        assertEquals(input, out,
                "Boards with all-unique positions should pass unchanged");
    }

    @Test
    void testDuplicateLetterInvalid() {
        var input = List.of("Aa", "bb", "cC");
        var out = filter.filter(input);
        assertTrue(out.isEmpty(),
                "Any duplicate letter (ignoring case) should filter out the board");
    }

    @Test
    void testDotCommaCollisionInvalid() {
        var input = List.of(".,", ",.", ".,");
        var out = filter.filter(input);
        assertTrue(out.isEmpty(),
                "Boards containing both '.' and ',' (same slot) should be invalid");
    }

    @Test
    void testMixedValidAndInvalid() {
        var input = List.of(
                "abc",   // valid
                "Aab",   // invalid: 'a' twice
                "A.B",   // valid: 'A', '.', 'B' all distinct
                ".X",    // valid
                ".X"     // valid (duplicates allowed here)
        );
        var expected = List.of("abc", "A.B", ".X", ".X");
        var out = filter.filter(input);
        assertEquals(expected, out,
                "Should keep only the valid boards, in original order");
    }
}
