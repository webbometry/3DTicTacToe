package precomputing.minimax;

import precomputing.minimax.kernels.Expansion;
import precomputing.minimax.kernels.CheckWin;
import precomputing.minimax.kernels.Canonicalization;

import java.io.IOException;
import java.nio.file.*;
import java.time.Duration;
import java.time.Instant;
import java.util.List;

public class Main {
    // maximum number of moves in 3×3×3 tic-tac-toe
    private static final int MAX_DEPTH = 27;

    public static void main(String[] args) {
        try {
            // Initialize pipeline components
            Canonicalization canon = new Canonicalization(
                    Paths.get("src/main/data/rotationMaps.txt")
            );
            Expansion expansion = new Expansion();
            CheckWin checkwin = new CheckWin();

            // Prepare output file for all terminal boards
            Path endFile = Paths.get("endPositions.txt");
            Files.deleteIfExists(endFile);
            Files.createFile(endFile);

            // Start from the empty board
            List<String> boards = List.of("");

            for (int step = 0; step < MAX_DEPTH; step++) {
                int nextDepth = step + 1; // after expansion, boards have length nextDepth
                System.out.printf("=== Step %2d (depth=%d) ===%n", nextDepth, nextDepth);

                // 1) Expansion
                Instant t0 = Instant.now();
                List<String> expanded = expansion.expandAll(boards);
                Instant t1 = Instant.now();
                long expMs = Duration.between(t0, t1).toMillis();
                System.out.printf(
                        "Expansion: %5d boards in %d ms%n",
                        expanded.size(), expMs
                );

                // 2) Canonicalization
                t0 = Instant.now();
                List<String> reduced = canon.canonicalize(expanded, nextDepth);
                t1 = Instant.now();
                long canonMs = Duration.between(t0, t1).toMillis();
                System.out.printf(
                        "Canonical:  %5d → %5d  (%,d reduced) in %d ms%n",
                        expanded.size(), reduced.size(), expanded.size() - reduced.size(), canonMs
                );

                if (step >= 8) {
                    // 3) Check for terminal positions
                    t0 = Instant.now();
                    CheckWin.Result result = checkwin.check(reduced, nextDepth);
                    t1 = Instant.now();
                    long checkMs = Duration.between(t0, t1).toMillis();
                    System.out.printf(
                            "CheckWin:    %5d → %5d terminals, %5d ongoing in %d ms%n",
                            reduced.size(), result.terminals.size(), result.nonTerminals.size(), checkMs
                    );

                    // 4) Append all new terminal boards
                    if (!result.terminals.isEmpty()) {
                        Files.write(endFile, result.terminals, StandardOpenOption.APPEND);
                    }

                    // 5) Prepare for next iteration
                    boards = result.nonTerminals;
                } else {
                    System.out.printf("%5d ongoing", reduced.size());
                    System.out.println();

                    // Prepare for next iteration
                    boards = reduced;
                }

                if (boards.isEmpty()) {
                    System.out.println("No ongoing boards remain. Stopping early.");
                    break;
                } else {
                    System.out.println();
                }
            }

            System.out.println("Done. All terminal positions (with ':') in " + endFile);
        } catch (IOException e) {
            System.err.println("I/O error: " + e.getMessage());
            e.printStackTrace();
        } catch (RuntimeException e) {
            System.err.println("Unexpected error: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
