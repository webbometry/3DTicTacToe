package precomputing.minimax;

import com.carrotsearch.hppc.LongArrayList;
import com.carrotsearch.hppc.cursors.LongCursor;
import support.CLContext;

import java.io.IOException;

public class GPUTimer {
    private static final int MAX_DEPTH = 27;
    private static final long RAM_BUDGET_BYTES = 40L * 1024 * 1024 * 1024; // 40 GB
    private static final int EXPANSION_FACTOR = 27;

    private final CLContext clContext;
    private final ExpandAndClassify expander;
    private final int maxBoardsPerBatch;

    // Statistics
    private long totalBoardsGenerated = 0;
    private long totalTerminalBoards = 0;
    private long totalGPUTime = 0;
    private long totalDepthTime = 0;

    public GPUTimer() throws IOException {
        this.clContext = new CLContext("cl/expand_and_classify.cl");

        // Calculate safe batch size based on GPU memory constraints
        long gpuMemoryBytes = clContext.maxAllocBytes;
        long memoryPerBoard = Long.BYTES;
        long memoryMultiplier = 1 + (4 * EXPANSION_FACTOR); // input + 4 output buffers

        int gpuLimitedBatch = (int) (gpuMemoryBytes / (memoryPerBoard * memoryMultiplier));
        int ramLimitedBatch = (int) (RAM_BUDGET_BYTES / (memoryPerBoard * EXPANSION_FACTOR));

        this.maxBoardsPerBatch = Math.min(gpuLimitedBatch, ramLimitedBatch);
        this.expander = new ExpandAndClassify(clContext, maxBoardsPerBatch);

        System.out.println("=== GPU Timer - Board Generation Performance Test ===");
        System.out.printf("GPU Memory Limit: %,d boards/batch%n", gpuLimitedBatch);
        System.out.printf("RAM Budget Limit: %,d boards/batch%n", ramLimitedBatch);
        System.out.printf("Batch Size: %,d boards%n", maxBoardsPerBatch);
        System.out.printf("GPU Max Alloc: %.2f MB%n", clContext.maxAllocBytes / (1024.0 * 1024.0));
        System.out.println();
    }

    public void runTimingTest() {
        long testStartTime = System.currentTimeMillis();

        // Start with empty board
        LongArrayList frontier = new LongArrayList();
        frontier.add(0L);

        for (int depth = 1; depth <= MAX_DEPTH; depth++) {
            long depthStartTime = System.currentTimeMillis();

            System.out.printf("┌─ Depth %d ─ Frontier: %,d boards ─────────────────────────────┐%n", depth, frontier.size());

            LongArrayList nextFrontier = new LongArrayList();
            long depthTerminals = 0;
            long depthGPUTime = 0;

            // Estimate memory usage for this depth
            long estimatedNextFrontierSize = frontier.size() * EXPANSION_FACTOR;
            long estimatedMemoryUsage = estimatedNextFrontierSize * Long.BYTES;

            System.out.printf("│ Estimated next frontier: %,d boards (%.1f MB)%n",
                estimatedNextFrontierSize, estimatedMemoryUsage / (1024.0 * 1024.0));

            int totalBatches = (frontier.size() + maxBoardsPerBatch - 1) / maxBoardsPerBatch;

            // Process frontier in batches
            for (int batchStart = 0; batchStart < frontier.size(); batchStart += maxBoardsPerBatch) {
                int batchEnd = Math.min(batchStart + maxBoardsPerBatch, frontier.size());
                int batchLen = batchEnd - batchStart;
                int batchNum = (batchStart / maxBoardsPerBatch) + 1;

                // Create batch
                LongArrayList batch = new LongArrayList(batchLen);
                for (int i = batchStart; i < batchEnd; i++) {
                    batch.add(frontier.get(i));
                }

                // Time GPU expansion
                long gpuStartTime = System.nanoTime();
                ExpandAndClassify.Result result = expander.run(batch, depth);
                long gpuEndTime = System.nanoTime();

                long batchGPUTime = (gpuEndTime - gpuStartTime) / 1_000_000; // Convert to milliseconds
                depthGPUTime += batchGPUTime;

                // Count terminals
                long batchTerminals = result.termX.size() + result.termO.size() + result.termTie.size();
                depthTerminals += batchTerminals;

                // Handle frontier expansion - collect all new frontier boards
                int totalNewFrontierBoards = result.frontierChunks.size();
                for (int i = 0; i < result.frontierChunks.size(); i++) {
                    long board = result.frontierChunks.get(i);
                    nextFrontier.add(board);
                }

                totalBoardsGenerated += batchTerminals + totalNewFrontierBoards;

                System.out.printf("│ Batch %d/%d: %,d→%,d terminals, %,d frontier (%dms)%n",
                    batchNum, totalBatches, batchLen, batchTerminals, totalNewFrontierBoards, batchGPUTime);

                // Show terminal breakdown for significant findings or early batches
                if (batchTerminals > 1000 || (batchTerminals > 0 && depth <= 10)) {
                    System.out.printf("│   Terminals: X=%,d, O=%,d, Tie=%,d%n",
                        result.termX.size(), result.termO.size(), result.termTie.size());
                }

                // Force garbage collection periodically
                if (batchNum % 10 == 0) {
                    System.gc();
                }
            }

            long depthEndTime = System.currentTimeMillis();
            long depthTotalTime = depthEndTime - depthStartTime;

            totalTerminalBoards += depthTerminals;
            totalGPUTime += depthGPUTime;
            totalDepthTime += depthTotalTime;

            System.out.printf("│ COMPLETE: %,d terminals found, %,d next frontier%n", depthTerminals, nextFrontier.size());
            System.out.printf("│ Timing: GPU %,dms (%.1f%%), Total %,dms%n",
                depthGPUTime, 
                depthTotalTime > 0 ? (100.0 * depthGPUTime / depthTotalTime) : 0.0,
                depthTotalTime);

            // Debug warning if no terminals found at late depths
            if (depthTerminals == 0 && depth >= 9) {
                System.out.printf("│ ⚠️  WARNING: No terminals at depth %d - check win detection!%n", depth);
            }

            System.out.println("└──────────────────────────────────────────────────────────────┘");
            System.out.println();

            // Clear the old frontier to free memory immediately
            frontier.clear();
            frontier = null;
            System.gc();

            // Update frontier for next iteration
            frontier = nextFrontier;

            // Break if no more boards to expand
            if (frontier.isEmpty()) {
                System.out.printf("🏁 Game tree complete at depth %d - no more boards to expand%n%n", depth);
                break;
            }
        }

        long testEndTime = System.currentTimeMillis();
        long totalTestTime = testEndTime - testStartTime;

        printSummary(totalTestTime);
    }

    private void printSummary(long totalTestTime) {
        System.out.println("┌─ PERFORMANCE SUMMARY ──────────────────────────────────────────┐");
        System.out.printf("│ Total Time: %,d ms (%.2f seconds)%n", totalTestTime, totalTestTime / 1000.0);
        System.out.printf("│ GPU Time:   %,d ms (%.1f%% utilization)%n", 
            totalGPUTime, totalTestTime > 0 ? (100.0 * totalGPUTime / totalTestTime) : 0.0);
        System.out.println("├─────────────────────────────────────────────────────────────────┤");
        System.out.printf("│ Boards Generated: %,d total%n", totalBoardsGenerated);
        System.out.printf("│ Terminal Boards:  %,d (%.1f%%)%n", 
            totalTerminalBoards, totalBoardsGenerated > 0 ? (100.0 * totalTerminalBoards / totalBoardsGenerated) : 0.0);
        System.out.printf("│ Frontier Boards:  %,d (%.1f%%)%n", 
            totalBoardsGenerated - totalTerminalBoards,
            totalBoardsGenerated > 0 ? (100.0 * (totalBoardsGenerated - totalTerminalBoards) / totalBoardsGenerated) : 0.0);
        System.out.println("├─────────────────────────────────────────────────────────────────┤");
        
        if (totalGPUTime > 0) {
            System.out.printf("│ GPU Throughput:     %,.0f boards/second%n", 
                (totalBoardsGenerated * 1000.0) / totalGPUTime);
        }
        
        if (totalTestTime > 0) {
            System.out.printf("│ Overall Throughput: %,.0f boards/second%n", 
                (totalBoardsGenerated * 1000.0) / totalTestTime);
        }

        // Add warning if no terminals found
        if (totalTerminalBoards == 0) {
            System.out.println("├─────────────────────────────────────────────────────────────────┤");
            System.out.println("│ ⚠️  CRITICAL WARNING: NO TERMINAL BOARDS FOUND!");
            System.out.println("│ Possible issues:");
            System.out.println("│ • WIN_MASKS configuration (check winLines.txt)");
            System.out.println("│ • Win detection logic in OpenCL kernel");
            System.out.println("│ • Board representation or bit manipulation");
            System.out.println("│ • 3D diagonal exclusion logic");
        }

        System.out.println("└─────────────────────────────────────────────────────────────────┘");
    }

    public static void main(String[] args) {
        try {
            GPUTimer timer = new GPUTimer();
            timer.runTimingTest();
        } catch (IOException e) {
            System.err.println("❌ Failed to initialize GPU context: " + e.getMessage());
            e.printStackTrace();
        } catch (Exception e) {
            System.err.println("❌ Error during timing test: " + e.getMessage());
            e.printStackTrace();
        }
    }
}