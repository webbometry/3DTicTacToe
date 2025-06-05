package precomputing.minimax;

import precomputing.minimax.kernels.*;
import precomputing.minimax.kernels.canonicalization.Canonicalization;

import java.io.IOException;
import java.nio.file.*;
import java.time.Duration;
import java.time.Instant;
import java.util.*;

public class Main {
    private static final int MAX_DEPTH = 27;
    private static int runningTotal = 0;

    public static void main(String[] args) {
        try {
            Canonicalization canon = new Canonicalization(
                    Paths.get("src/main/data/rotationMaps.txt")
            );
            Expansion expansion = new Expansion();
            CheckWin checkwin = new CheckWin();

            Path endFile = Paths.get("endPositions.txt");
            Files.deleteIfExists(endFile);
            Files.createFile(endFile);

            List<String> boards = List.of("");

            for (int step = 0; step < MAX_DEPTH; step++) {
                if (step != 0) {
                    System.out.printf("Total runtime: %d ms%n%n", runningTotal);
                }

                int nextDepth = step + 1;
                System.out.printf("=== Step %2d (depth=%d) ===%n", nextDepth, nextDepth);

                long MAX_MEMORY_BYTES = 60L * 1024 * 1024 * 1024; // 60 GB
                int EST_BYTES_PER_BOARD = 100;
                int totalBoardsCount = boards.size();
                long estMemory = (long) totalBoardsCount * EST_BYTES_PER_BOARD;
                int batches = (int) Math.ceil((double) estMemory / MAX_MEMORY_BYTES);
                batches = Math.max(batches, 1);

                System.out.printf(
                        "Estimated memory: %.2f GB (%d boards). Splitting into %d batch(es)...%n",
                        estMemory / 1e9, totalBoardsCount, batches
                );

                Instant depthStart = Instant.now();
                Set<String> allReducedSet = new HashSet<>();

                for (int bi = 0; bi < batches; bi++) {
                    int fromIndex = bi * totalBoardsCount / batches;
                    int toIndex = (bi + 1) * totalBoardsCount / batches;
                    List<String> batchBoards = boards.subList(fromIndex, toIndex);

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

                    allReducedSet.addAll(reducedBatch);

                    expandedBatch = null;
                    reducedBatch = null;
                    System.gc();
                }

                Instant depthEnd = Instant.now();
                long depthMs = Duration.between(depthStart, depthEnd).toMillis();
                System.out.printf(" Total this depth (all batches): %d ms%n", depthMs);

                List<String> reduced = List.copyOf(allReducedSet);

                if (step >= 8) {
                    System.out.println("=== CheckWin Phase ===");

                    int checkTotal = reduced.size();
                    long checkEstMem = (long) checkTotal * EST_BYTES_PER_BOARD;
                    int checkBatches = (int) Math.ceil((double) checkEstMem / MAX_MEMORY_BYTES);
                    checkBatches = Math.max(checkBatches, 1);

                    System.out.printf(
                            "Estimated CheckWin memory: %.2f GB (%d boards). Splitting into %d batch(es)...%n",
                            checkEstMem / 1e9, checkTotal, checkBatches
                    );

                    List<String> terminals = new ArrayList<>();
                    List<String> nonTerminals = new ArrayList<>();

                    for (int i = 0; i < checkBatches; i++) {
                        int from = i * checkTotal / checkBatches;
                        int to = (i + 1) * checkTotal / checkBatches;
                        List<String> checkBatch = reduced.subList(from, to);

                        Instant cbStart = Instant.now();
                        CheckWin.Result partial = checkwin.check(checkBatch, nextDepth);
                        Instant cbEnd = Instant.now();

                        terminals.addAll(partial.terminals);
                        nonTerminals.addAll(partial.nonTerminals);

                        long cbMs = Duration.between(cbStart, cbEnd).toMillis();
                        System.out.printf(" CheckWin batch %d/%d: %d → %d terminals, %d ongoing in %d ms%n",
                                i + 1, checkBatches, checkBatch.size(),
                                partial.terminals.size(), partial.nonTerminals.size(), cbMs);

                        runningTotal += cbMs;

                        checkBatch = null;
                        partial = null;
                        System.gc();
                    }

                    if (!terminals.isEmpty()) {
                        Files.write(endFile, terminals, StandardOpenOption.APPEND);
                    }

                    boards = nonTerminals;
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
