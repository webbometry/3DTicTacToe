package precomputing.minimax;

import precomputing.minimax.kernels.*;
import precomputing.minimax.kernels.canonicalization.Canonicalization;

import java.io.IOException;
import java.nio.file.*;
import java.time.Duration;
import java.time.Instant;
import java.util.List;
import java.util.Set;
import java.util.HashSet;

public class Main {
    // maximum number of moves in 3×3×3 tic-tac-toe
    private static final int MAX_DEPTH = 27;
    private static int runningTotal = 0;

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
                if (step != 0) {
                    System.out.printf("Total runtime: %d ms%n", runningTotal);
                    System.out.println();
                }

                int nextDepth = step + 1; // after expansion, boards have length nextDepth
                System.out.printf("=== Step %2d (depth=%d) ===%n", nextDepth, nextDepth);

                //
                // ==== Batching Logic Begins Here ====
                //
                // Estimate how many boards we can handle in one batch without exceeding 60 GB
                long MAX_MEMORY_BYTES = 60L * 1024 * 1024 * 1024; // 60 GB
                int EST_BYTES_PER_BOARD = 100; // rough per-String estimate (~100 bytes)
                int totalBoardsCount = boards.size();
                long estMemory = (long) totalBoardsCount * EST_BYTES_PER_BOARD;
                int batches = (int) Math.ceil((double) estMemory / MAX_MEMORY_BYTES);
                batches = Math.max(batches, 1);

                System.out.printf(
                        "Estimated memory: %.2f GB (%d boards). Splitting into %d batch(es)...%n",
                        estMemory / 1e9, totalBoardsCount, batches
                );

                Instant depthStart = Instant.now();

                // We accumulate all reduced boards into this set, to dedupe across batches
                Set<String> allReducedSet = new HashSet<>();

                for (int bi = 0; bi < batches; bi++) {
                    int fromIndex = bi * totalBoardsCount / batches;
                    int toIndex = (bi + 1) * totalBoardsCount / batches;
                    List<String> batchBoards = boards.subList(fromIndex, toIndex);

                    // 1a) Expand this batch
                    Instant bt0 = Instant.now();
                    List<String> expandedBatch = expansion.expandAll(batchBoards);
                    Instant bt1 = Instant.now();
                    long bexpMs = Duration.between(bt0, bt1).toMillis();
                    System.out.printf(
                            " Batch %d/%d Expansion: %5d → %5d boards in %d ms%n",
                            bi + 1, batches,
                            batchBoards.size(), expandedBatch.size(), bexpMs
                    );
                    runningTotal += bexpMs;

                    // 2a) Canonicalize this batch
                    Instant bc0 = Instant.now();
                    Canonicalization.Result bRes = canon.canonicalize(expandedBatch, nextDepth);
                    List<String> reducedBatch = bRes.boards();
                    Instant bc1 = Instant.now();
                    long bcanonMs = Duration.between(bc0, bc1).toMillis();
                    System.out.printf(
                            " Batch %d/%d Canonical:  %5d → %5d (invalid: %,4d, dupes: %,4d, rotations: %,4d, total removed: %,4d) in %d ms%n",
                            bi + 1, batches,
                            expandedBatch.size(), reducedBatch.size(),
                            bRes.removedInvalid(), bRes.removedDuplicates(), bRes.removedRotations(),
                            (bRes.removedInvalid() + bRes.removedDuplicates() + bRes.removedRotations()),
                            bcanonMs
                    );
                    runningTotal += bcanonMs;

                    // Accumulate into the global set, deduplicating across batches
                    allReducedSet.addAll(reducedBatch);

                    // Help GC
                    expandedBatch = null;
                    reducedBatch = null;
                    System.gc();
                }

                Instant depthEnd = Instant.now();
                long depthMs = Duration.between(depthStart, depthEnd).toMillis();
                System.out.printf(" Total this depth (all batches): %d ms%n", depthMs);

                // Convert set of all reduced boards into a List for the next steps
                List<String> reduced = List.copyOf(allReducedSet);

                //
                // ==== Batching Logic Ends Here ====
                //

                if (step >= 8) {
                    // 3) Check for terminal positions
                    Instant t0 = Instant.now();
                    CheckWin.Result result = checkwin.check(reduced, nextDepth);
                    Instant t1 = Instant.now();
                    long checkMs = Duration.between(t0, t1).toMillis();
                    System.out.printf(
                            "CheckWin:    %5d → %5d terminals, %5d ongoing in %d ms%n",
                            reduced.size(), result.terminals.size(), result.nonTerminals.size(), checkMs
                    );
                    runningTotal += checkMs;

                    // 4) Append all new terminal boards
                    if (!result.terminals.isEmpty()) {
                        Files.write(endFile, result.terminals, StandardOpenOption.APPEND);
                    }

                    // 5) Prepare for next iteration
                    boards = result.nonTerminals;
                } else {
                    System.out.printf("%5d ongoing%n", reduced.size());
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
