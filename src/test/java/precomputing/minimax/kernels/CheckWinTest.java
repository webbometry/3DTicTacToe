package precomputing.minimax.kernels;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class CheckWinTest {
    private static CheckWin checker;

    @BeforeAll
    static void setUp() {
        checker = new CheckWin();
    }

    /**
     * For step ≤ 8, everything is considered non-terminal.
     */
    @Test
    void testEarlyExitNonTerminal() {
        List<String> boards = List.of("ABC", "abc", ".........");
        int step = 8;  // ≤ EARLY_THRESHOLD
        CheckWin.Result result = checker.check(boards, step);

        assertTrue(result.terminals.isEmpty(),
                "No terminals at or below threshold");
        assertEquals(boards, result.nonTerminals,
                "Non-terminals must equal input when step ≤ threshold");
    }

    /**
     * At step > 8, a board containing an X-winning line must be marked terminal.
     */
    @Test
    void testDetectXWin() {
        List<String> boards = List.of("ABC");
        int step = 9;  // > EARLY_THRESHOLD
        CheckWin.Result result = checker.check(boards, step);

        assertEquals(List.of("ABC:"), result.terminals,
                "Board with X win must appear in terminals with ':'");
        assertTrue(result.nonTerminals.isEmpty(),
                "No non-terminals when X has already won");
    }

    /**
     * At step > 8, a board containing an O-winning line must be marked terminal.
     */
    @Test
    void testDetectOWin() {
        List<String> boards = List.of("abc");
        int step = 9;
        CheckWin.Result result = checker.check(boards, step);

        assertEquals(List.of("abc:"), result.terminals,
                "Board with O win must appear in terminals with ':'");
        assertTrue(result.nonTerminals.isEmpty(),
                "No non-terminals when O has already won");
    }

    /**
     * A full board (step == MAX_DEPTH) with no wins should still be terminal (draw).
     * Using 27 '.' characters to avoid false-positive wins.
     */
    @Test
    void testDetectDrawOnFullBoard() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 27; i++) sb.append('.');
        String fullBoard = sb.toString();
        int step = 27;  // MAX_DEPTH

        CheckWin.Result result = checker.check(List.of(fullBoard), step);

        assertEquals(1, result.terminals.size(),
                "Full board must be reported as terminal (draw)");
        assertEquals(fullBoard + ":", result.terminals.get(0),
                "Draw board must have ':' appended");
        assertTrue(result.nonTerminals.isEmpty(),
                "No non-terminals for a full board");
    }

    /**
     * At step > 8, a board with no winning lines and not full should be non-terminal.
     */
    @Test
    void testNonTerminalAfterThreshold() {
        // 'ACE' has no ABC, DEF, etc. win-lines
        String board = "ACE";
        int step = 9;

        CheckWin.Result result = checker.check(List.of(board), step);

        assertTrue(result.terminals.isEmpty(),
                "Board with no win and not full must not be terminal");
        assertEquals(List.of(board), result.nonTerminals,
                "Non-terminal board should appear unchanged");
    }

    /**
     * Mixed input: some boards terminal, others not, all at the same step.
     */
    @Test
    void testMixedBoardsPartition() {
        // "ABC" → X win; "abc" → O win; "ACE" → no win
        List<String> boards = List.of("ABC", "abc", "ACE");
        int step = 9;

        CheckWin.Result result = checker.check(boards, step);

        // Expect two terminals: ABC: and abc:
        assertEquals(2, result.terminals.size(),
                "Two boards should be terminal");
        assertTrue(result.terminals.contains("ABC:"));
        assertTrue(result.terminals.contains("abc:"));

        // Expect one non-terminal: ACE
        assertEquals(1, result.nonTerminals.size(),
                "Exactly one non-terminal board should remain");
        assertEquals("ACE", result.nonTerminals.get(0));
    }
}