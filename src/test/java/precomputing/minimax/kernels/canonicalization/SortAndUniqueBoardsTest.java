// src/test/java/precomputing/minimax/kernels/canonicalization/SortAndUniqueBoardsTest.java
package precomputing.minimax.kernels.canonicalization;

import org.jocl.CL;
import org.junit.jupiter.api.*;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class SortAndUniqueBoardsTest {
    private static SortAndUniqueBoards sorter;

//    @BeforeAll
//    static void setUp() {
//        CL.setExceptionsEnabled(true);
//        sorter = new SortAndUniqueBoards();
//    }
//
//    @AfterAll
//    static void tearDown() {
//        sorter.release();
//    }
//
//    @Test
//    void testEmpty() {
//        assertTrue(sorter.sortAndUnique(List.of()).isEmpty(),
//                "Empty input → empty output");
//    }
//
//    @Test
//    void testSingleBoardSorted() {
//        var in  = List.of("DaBc");
//        var out = sorter.sortAndUnique(in);
//        assertEquals(List.of("aBcD"), out,
//                "\"DaBc\" should sort (case-insensitive) to \"aBcD\"");
//    }
//
//    @Test
//    void testAlreadySortedUnchanged() {
//        var in  = List.of("aBcD");
//        var out = sorter.sortAndUnique(in);
//        assertEquals(List.of("aBcD"), out,
//                "Already sorted board remains unchanged");
//    }
//
//    @Test
//    void testCaseInsensitiveOrdering() {
//        var in       = List.of("BaC", "cBa");
//        var out      = sorter.sortAndUnique(in);
//        var expected = List.of("aBC", "aBc");
//        assertEquals(expected, out,
//                "\"BaC\"→\"aBC\", \"cBa\"→\"aBc\"");
//    }
//
//    @Test
//    void testExactDuplicatesRemoved() {
//        var in  = List.of("DaBc","aBcD","gem","Meg","gem");
//        var out = sorter.sortAndUnique(in);
//        var expected = List.of("aBcD","egm","egM");
//        assertEquals(expected, out,
//                "Duplicates (exact sorted matches) should be collapsed, first-seen kept");
//    }
}
