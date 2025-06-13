package precomputing.minimax;

import com.carrotsearch.hppc.LongArrayList;
import com.carrotsearch.hppc.cursors.LongCursor;
import support.CLContext;

import java.io.*;
import java.nio.file.*;
import java.util.*;

public class GPUTimer {
    private static final int MAX_DEPTH = 27;
    private static final long RAM_BUDGET_BYTES = 40L * 1024 * 1024 * 1024; // 40 GB
    private static final int EXPANSION_FACTOR = 27;
    private static final int MAX_FRONTIER_SIZE = 50_000_000; // Max boards in memory frontier
    private static final int FRONTIER_SPLIT_THRESHOLD = 30_000_000; // When to start splitting
    private static final String TEMP_DIR = "temp_frontiers";
    private static final String OUTPUT_DIR = "C:\\Users\\webbometric\\Documents\\GitHub\\3DTicTacToe\\src\\main\\resources\\MiniMax";

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

        // Create temp directory for frontier files
        Files.createDirectories(Paths.get(TEMP_DIR));
        // Create output directory for terminals
        Files.createDirectories(Paths.get(OUTPUT_DIR));

        System.out.println("=== GPU Timer - Board Generation Performance Test ===");
        System.out.printf("GPU Memory Limit: %,d boards/batch%n", gpuLimitedBatch);
        System.out.printf("RAM Budget Limit: %,d boards/batch%n", ramLimitedBatch);
        System.out.printf("Batch Size: %,d boards%n", maxBoardsPerBatch);
        System.out.printf("Max Frontier Size: %,d boards%n", MAX_FRONTIER_SIZE);
        System.out.printf("GPU Max Alloc: %.2f MB%n", clContext.maxAllocBytes / (1024.0 * 1024.0));
        System.out.printf("Output Directory: %s%n", OUTPUT_DIR);
        System.out.println();
    }

    // Multi-frontier management
    private static class MultiFrontier {
        private final List<LongArrayList> memoryFrontiers;
        private final List<String> diskFrontiers;
        private final Set<String> processedFiles; // Track files that have been fully consumed
        private long totalSize;

        public MultiFrontier() {
            this.memoryFrontiers = new ArrayList<>();
            this.diskFrontiers = new ArrayList<>();
            this.processedFiles = new HashSet<>();
            this.totalSize = 0;
        }

        public void addBoards(LongArrayList boards) throws IOException {
            if (boards.isEmpty()) return;

            // If we can fit in current memory frontier, do so
            if (!memoryFrontiers.isEmpty()) {
                LongArrayList current = memoryFrontiers.get(memoryFrontiers.size() - 1);
                if (current.size() + boards.size() <= MAX_FRONTIER_SIZE) {
                    for (int i = 0; i < boards.size(); i++) {
                        current.add(boards.get(i));
                    }
                    totalSize += boards.size();
                    return;
                }
            }

            // If current memory frontier is too big, move it to disk
            if (!memoryFrontiers.isEmpty()) {
                LongArrayList current = memoryFrontiers.get(memoryFrontiers.size() - 1);
                if (current.size() >= FRONTIER_SPLIT_THRESHOLD) {
                    String filename = TEMP_DIR + "/frontier_" + System.currentTimeMillis() + ".dat";
                    saveFrontierToDisk(current, filename);
                    diskFrontiers.add(filename);
                    memoryFrontiers.remove(memoryFrontiers.size() - 1);
                    current.clear();
                }
            }

            // Create new memory frontier
            LongArrayList newFrontier = new LongArrayList(Math.min(MAX_FRONTIER_SIZE, boards.size()));
            for (int i = 0; i < boards.size(); i++) {
                newFrontier.add(boards.get(i));
            }
            memoryFrontiers.add(newFrontier);
            totalSize += boards.size();
        }

        public long size() {
            return totalSize;
        }

        public boolean isEmpty() {
            return totalSize == 0;
        }

        public FrontierIterator iterator() {
            return new FrontierIterator(this);
        }

        public void clear() {
            // Clear memory frontiers
            for (LongArrayList frontier : memoryFrontiers) {
                frontier.clear();
            }
            memoryFrontiers.clear();

            // Delete disk frontiers
            for (String filename : diskFrontiers) {
                try {
                    Files.deleteIfExists(Paths.get(filename));
                } catch (IOException e) {
                    System.err.println("Warning: Could not delete " + filename);
                }
            }
            diskFrontiers.clear();
            processedFiles.clear();
            totalSize = 0;
        }

        public void cleanupProcessedFiles() {
            // Only delete files that have been marked as processed
            List<String> toRemove = new ArrayList<>();
            for (String filename : processedFiles) {
                try {
                    Files.deleteIfExists(Paths.get(filename));
                    toRemove.add(filename);
                } catch (IOException e) {
                    // Ignore deletion errors
                }
            }
            processedFiles.removeAll(toRemove);
        }

        private void markFileAsProcessed(String filename) {
            processedFiles.add(filename);
        }

        private void saveFrontierToDisk(LongArrayList frontier, String filename) throws IOException {
            try (DataOutputStream dos = new DataOutputStream(new BufferedOutputStream(
                    new FileOutputStream(filename)))) {
                dos.writeInt(frontier.size());
                for (int i = 0; i < frontier.size(); i++) {
                    dos.writeLong(frontier.get(i));
                }
            }
            System.out.printf("│ 💾 Saved %,d boards to disk: %s%n", frontier.size(), filename);
        }

        private LongArrayList loadFrontierFromDisk(String filename) throws IOException {
            try (DataInputStream dis = new DataInputStream(new BufferedInputStream(
                    new FileInputStream(filename)))) {
                int size = dis.readInt();
                LongArrayList frontier = new LongArrayList(size);
                for (int i = 0; i < size; i++) {
                    frontier.add(dis.readLong());
                }
                return frontier;
            }
        }
    }

    private static class FrontierIterator {
        private final MultiFrontier multiFrontier;
        private int memoryIndex;
        private int diskIndex;
        private int positionInCurrent;
        private LongArrayList currentMemoryFrontier;
        private LongArrayList currentDiskFrontier;
        private String currentDiskFile;

        public FrontierIterator(MultiFrontier multiFrontier) {
            this.multiFrontier = multiFrontier;
            this.memoryIndex = 0;
            this.diskIndex = 0;
            this.positionInCurrent = 0;
            this.currentDiskFile = null;
            loadNextFrontier();
        }

        private void loadNextFrontier() {
            // Try memory frontiers first
            if (memoryIndex < multiFrontier.memoryFrontiers.size()) {
                currentMemoryFrontier = multiFrontier.memoryFrontiers.get(memoryIndex);
                currentDiskFrontier = null;
                currentDiskFile = null;
                memoryIndex++;
                positionInCurrent = 0;
                return;
            }

            // Then disk frontiers
            if (diskIndex < multiFrontier.diskFrontiers.size()) {
                try {
                    String filename = multiFrontier.diskFrontiers.get(diskIndex);
                    currentDiskFrontier = multiFrontier.loadFrontierFromDisk(filename);
                    currentMemoryFrontier = null;
                    currentDiskFile = filename;
                    diskIndex++;
                    positionInCurrent = 0;
                    System.out.printf("│ 📀 Loaded %,d boards from disk%n", currentDiskFrontier.size());
                } catch (IOException e) {
                    System.err.println("Error loading frontier from disk: " + e.getMessage());
                    currentDiskFrontier = null;
                    currentMemoryFrontier = null;
                    currentDiskFile = null;
                }
                return;
            }

            // No more frontiers
            currentMemoryFrontier = null;
            currentDiskFrontier = null;
            currentDiskFile = null;
        }

        public LongArrayList getNextChunk(int maxSize) {
            LongArrayList chunk = new LongArrayList();

            while (chunk.size() < maxSize && hasMore()) {
                LongArrayList current = (currentMemoryFrontier != null) ? currentMemoryFrontier : currentDiskFrontier;
                
                if (current == null || positionInCurrent >= current.size()) {
                    // Mark current disk file as fully processed before cleaning up
                    if (currentDiskFrontier != null && currentDiskFile != null) {
                        multiFrontier.markFileAsProcessed(currentDiskFile);
                        currentDiskFrontier.clear();
                        currentDiskFrontier = null;
                        currentDiskFile = null;
                    }
                    loadNextFrontier();
                    continue;
                }

                int remaining = Math.min(maxSize - chunk.size(), current.size() - positionInCurrent);
                for (int i = 0; i < remaining; i++) {
                    chunk.add(current.get(positionInCurrent + i));
                }
                positionInCurrent += remaining;
            }

            return chunk;
        }

        public boolean hasMore() {
            if (currentMemoryFrontier != null && positionInCurrent < currentMemoryFrontier.size()) {
                return true;
            }
            if (currentDiskFrontier != null && positionInCurrent < currentDiskFrontier.size()) {
                return true;
            }
            return memoryIndex < multiFrontier.memoryFrontiers.size() || 
                   diskIndex < multiFrontier.diskFrontiers.size();
        }
    }

    private void saveTerminalsToFile(LongArrayList terminals, String filename) throws IOException {
        if (terminals.isEmpty()) return;
        
        try (DataOutputStream dos = new DataOutputStream(new BufferedOutputStream(
                new FileOutputStream(filename, true)))) { // append mode
            for (int i = 0; i < terminals.size(); i++) {
                dos.writeLong(terminals.get(i));
            }
        }
    }

    public void runTimingTest() {
        long testStartTime = System.currentTimeMillis();

        try {
            // Start with empty board
            MultiFrontier frontier = new MultiFrontier();
            LongArrayList initialBoard = new LongArrayList();
            initialBoard.add(0L);
            frontier.addBoards(initialBoard);

            for (int depth = 1; depth <= MAX_DEPTH; depth++) {
                long depthStartTime = System.currentTimeMillis();

                System.out.printf("┌─ Depth %d ─ Frontier: %,d boards ─────────────────────────────┐%n", depth, frontier.size());

                // Process depth using multi-frontier system
                MultiFrontier nextFrontier = processDepthWithMultiFrontier(frontier, depth);

                long depthEndTime = System.currentTimeMillis();
                long depthTotalTime = depthEndTime - depthStartTime;

                System.out.printf("│ COMPLETE: Next frontier size: %,d%n", nextFrontier.size());
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

            // Clean up final frontier
            if (frontier != null) {
                frontier.clear();
            }

        } catch (IOException e) {
            System.err.println("❌ Error managing frontiers: " + e.getMessage());
            e.printStackTrace();
        } finally {
            // Clean up temp directory
            try {
                Files.walk(Paths.get(TEMP_DIR))
                     .sorted(Comparator.reverseOrder())
                     .map(Path::toFile)
                     .forEach(File::delete);
            } catch (IOException e) {
                System.err.println("Warning: Could not clean up temp directory");
            }
        }

        long testEndTime = System.currentTimeMillis();
        long totalTestTime = testEndTime - testStartTime;

        printSummary(totalTestTime);
    }

    private MultiFrontier processDepthWithMultiFrontier(MultiFrontier frontier, int depth) throws IOException {
        // Calculate safe chunk size
        long memoryPerBoard = Long.BYTES * EXPANSION_FACTOR;
        int safeChunkSize = (int) Math.min(maxBoardsPerBatch, 
            Math.min(FRONTIER_SPLIT_THRESHOLD / 2, RAM_BUDGET_BYTES / (memoryPerBoard * 8)));

        // Calculate total chunks for progress tracking
        int totalChunks = (int) ((frontier.size() + safeChunkSize - 1) / safeChunkSize);

        MultiFrontier nextFrontier = new MultiFrontier();
        long depthTerminals = 0;
        long depthGPUTime = 0;

        // Initialize terminal files for this depth
        String termXFile = OUTPUT_DIR + "/terminals_depth" + depth + "_X.dat";
        String termOFile = OUTPUT_DIR + "/terminals_depth" + depth + "_O.dat";
        String termTieFile = OUTPUT_DIR + "/terminals_depth" + depth + "_TIE.dat";

        // Clear existing files
        Files.deleteIfExists(Paths.get(termXFile));
        Files.deleteIfExists(Paths.get(termOFile));
        Files.deleteIfExists(Paths.get(termTieFile));

        FrontierIterator iterator = frontier.iterator();
        int chunkNum = 0;

        while (iterator.hasMore()) {
            chunkNum++;
            
            // Get next chunk of boards to process
            LongArrayList chunk = iterator.getNextChunk(safeChunkSize);
            if (chunk.isEmpty()) break;

            System.out.printf("│ Processing chunk %d/%d: %,d boards%n", chunkNum, totalChunks, chunk.size());

            // Monitor memory
            Runtime runtime = Runtime.getRuntime();
            long usedMemory = runtime.totalMemory() - runtime.freeMemory();
            long maxMemory = runtime.maxMemory();
            double memoryUsage = (double) usedMemory / maxMemory;

            if (memoryUsage > 0.8) {
                System.out.printf("│   Memory: %.1f%% - forcing GC%n", memoryUsage * 100);
                System.gc();
                Thread.yield();
            }

            // Process chunk in batches
            LongArrayList chunkResults = new LongArrayList();
            long chunkTerminals = 0;
            long chunkGPUTime = 0;

            int totalBatches = (chunk.size() + maxBoardsPerBatch - 1) / maxBoardsPerBatch;

            for (int batchStart = 0; batchStart < chunk.size(); batchStart += maxBoardsPerBatch) {
                int batchEnd = Math.min(batchStart + maxBoardsPerBatch, chunk.size());
                int batchLen = batchEnd - batchStart;
                int batchNum = (batchStart / maxBoardsPerBatch) + 1;

                // Create batch
                LongArrayList batch = new LongArrayList(batchLen);
                for (int i = batchStart; i < batchEnd; i++) {
                    batch.add(chunk.get(i));
                }

                // Time GPU expansion
                long gpuStartTime = System.nanoTime();
                ExpandAndClassify.Result result = expander.run(batch, depth);
                long gpuEndTime = System.nanoTime();

                long batchGPUTime = (gpuEndTime - gpuStartTime) / 1_000_000;
                chunkGPUTime += batchGPUTime;

                // Count terminals and save to files
                long batchTerminals = result.termX.size() + result.termO.size() + result.termTie.size();
                chunkTerminals += batchTerminals;

                // Save terminals to files
                saveTerminalsToFile(result.termX, termXFile);
                saveTerminalsToFile(result.termO, termOFile);
                saveTerminalsToFile(result.termTie, termTieFile);

                // Collect frontier boards from this batch
                for (int i = 0; i < result.frontierChunks.size(); i++) {
                    chunkResults.add(result.frontierChunks.get(i));
                }

                totalBoardsGenerated += batchTerminals + result.frontierChunks.size();

                if (batchNum % 5 == 0 || totalBatches <= 10) {
                    System.out.printf("│   Batch %d/%d: %,d→%,d terminals, %,d frontier (%dms)%n",
                        batchNum, totalBatches, batchLen, batchTerminals, result.frontierChunks.size(), batchGPUTime);
                }

                // Clear batch immediately
                batch.clear();
                batch = null;

                // Add results to next frontier in smaller chunks to avoid memory issues
                if (chunkResults.size() > FRONTIER_SPLIT_THRESHOLD / 4) {
                    nextFrontier.addBoards(chunkResults);
                    chunkResults.clear();
                    chunkResults = new LongArrayList();
                }
            }

            // Add remaining results
            if (!chunkResults.isEmpty()) {
                nextFrontier.addBoards(chunkResults);
                chunkResults.clear();
            }

            depthTerminals += chunkTerminals;
            depthGPUTime += chunkGPUTime;

            System.out.printf("│ Chunk %d/%d complete: %,d terminals, next frontier: %,d%n",
                chunkNum, totalChunks, chunkTerminals, nextFrontier.size());

            // Clear chunk
            chunk.clear();
            chunk = null;

            // Force GC every few chunks and clean up processed temp frontiers
            if (chunkNum % 3 == 0) {
                System.gc();
                frontier.cleanupProcessedFiles(); // Only delete files that have been fully consumed
            }
        }

        totalTerminalBoards += depthTerminals;
        totalGPUTime += depthGPUTime;

        System.out.printf("│ Depth Summary: %,d terminals, %,d next frontier%n", depthTerminals, nextFrontier.size());
        System.out.printf("│ Terminal files saved to: %s%n", OUTPUT_DIR);

        if (depthTerminals == 0 && depth >= 9) {
            System.out.printf("│ ⚠️  WARNING: No terminals at depth %d - check win detection!%n", depth);
        }

        return nextFrontier;
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
        System.out.printf("│ Terminal files saved to: %s%n", OUTPUT_DIR);
        
        if (totalGPUTime > 0) {
            System.out.printf("│ GPU Throughput:     %,.0f boards/second%n", 
                (totalBoardsGenerated * 1000.0) / totalGPUTime);
        }
        
        if (totalTestTime > 0) {
            System.out.printf("│ Overall Throughput: %,.0f boards/second%n", 
                (totalBoardsGenerated * 1000.0) / totalTestTime);
        }

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