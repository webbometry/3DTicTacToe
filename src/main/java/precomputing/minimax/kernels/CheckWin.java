package precomputing.minimax.kernels;

import precomputing.minimax.*;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class CheckWin {
    private static final int EARLY_THRESHOLD = 8;
    private static final int MAX_DEPTH = 27;

    private final List<char[]> lowerLines = new ArrayList<>();
    private final List<char[]> upperLines = new ArrayList<>();

    /**
     * Holds terminal vs ongoing partitions.
     */
    public static class Result {
        public final List<String> terminals;
        public final List<String> nonTerminals;

        public Result(List<String> t, List<String> n) {
            this.terminals = t;
            this.nonTerminals = n;
        }
    }

    /**
     * No‐arg ctor uses your data files under src/main/data.
     */
    public CheckWin() {
        this(
                Paths.get("src/main/data/winLinesLower.txt"),
                Paths.get("src/main/data/winLinesUpper.txt")
        );
    }

    /**
     * Load the two win‐line files into memory.
     */
    private CheckWin(Path lowerPath, Path upperPath) {
        try {
            for (String line : Files.readAllLines(lowerPath, StandardCharsets.UTF_8)) {
                String[] p = line.trim().split("\\s+");
                if (p.length == 3) {
                    lowerLines.add(new char[]{p[0].charAt(0), p[1].charAt(0), p[2].charAt(0)});
                }
            }
            for (String line : Files.readAllLines(upperPath, StandardCharsets.UTF_8)) {
                String[] p = line.trim().split("\\s+");
                if (p.length == 3) {
                    upperLines.add(new char[]{p[0].charAt(0), p[1].charAt(0), p[2].charAt(0)});
                }
            }
        } catch (IOException e) {
            throw new UncheckedIOException("Unable to load win-lines data", e);
        }
    }

    /**
     * Partition boards into terminals (appending ':') and nonTerminals.
     *
     * @param boards list of move-strings, each of length == step
     * @param step   the current depth (number of moves played)
     */
    public Result check(List<String> boards, int step) {
        List<String> terms = new ArrayList<>();
        List<String> nonTerms = new ArrayList<>();

        for (String b : boards) {
            boolean terminal = false;

            // 1) nothing can end before EARLY_THRESHOLD+1
            if (step <= EARLY_THRESHOLD) {
                terminal = false;

                // 2) full‐board draw
            } else if (step == MAX_DEPTH) {
                terminal = true;

                // 3) any single X-line
            } else {
                for (char[] tri : upperLines) {
                    if (b.indexOf(tri[0]) >= 0
                            && b.indexOf(tri[1]) >= 0
                            && b.indexOf(tri[2]) >= 0) {
                        terminal = true;
                        break;
                    }
                }
                // 4) any single O-line
                if (!terminal) {
                    for (char[] tri : lowerLines) {
                        if (b.indexOf(tri[0]) >= 0
                                && b.indexOf(tri[1]) >= 0
                                && b.indexOf(tri[2]) >= 0) {
                            terminal = true;
                            break;
                        }
                    }
                }
            }

            if (terminal) {
                terms.add(b + ":");
            } else {
                nonTerms.add(b);
            }
        }

        return new Result(terms, nonTerms);
    }
}