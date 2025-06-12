package precomputing.minimax;

import com.carrotsearch.hppc.LongArrayList;
import com.carrotsearch.hppc.cursors.LongCursor;
import support.CLContext;
import org.jocl.Sizeof;

import java.io.*;
import java.nio.LongBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.Paths;

public class Main {
    private static final int MAX_DEPTH = 27;
    static final long RAM_BUDGET_BYTES = 40L * 1024 * 1024 * 1024; // 40 GB
    static final int EXPANSION_FACTOR = 27;
    static int MAX_BOARDS_PER_BATCH = (int) (RAM_BUDGET_BYTES / (Long.BYTES * (1 + EXPANSION_FACTOR)));
    private static final long MAX_POSSIBLE_TERMS = 1_000_000_000L;

    public static void main(String[] args) throws IOException {
        try {
            // 1) Initialize OpenCL context & expander
            CLContext clContext = new CLContext("cl/expand_and_classify.cl");
            // Determine the maximum boards-per-chunk from device limits
            int maxBoards = (int) (clContext.maxAllocBytes / Sizeof.cl_ulong);
            ExpandAndClassify expander = new ExpandAndClassify(clContext, maxBoards);

            // 2) Prepare terminals.bin
            Path outFile = Paths.get("src/main/resources/MiniMax/terminals.bin");
            outFile.getParent().toFile().mkdirs();

            FileChannel fc = new RandomAccessFile(new File("C:\\Users\\webbometric\\Documents\\GitHub\\3DTicTacToe\\src\\main\\resources\\MiniMax\\terminals.bin"), "rw").getChannel();
            long maxBytes = MAX_POSSIBLE_TERMS * Long.BYTES;
            fc.truncate(maxBytes);
            LongBuffer lb = fc.map(FileChannel.MapMode.READ_WRITE, 0, fc.size()).asLongBuffer();

            // 3) Seed frontier with the empty board (bitboard = 0)
            LongArrayList frontier = new LongArrayList();
            frontier.add(0L);

            // 4) Iterate depths 1 through MAX_DEPTH
            for (int depth = 1; depth <= MAX_DEPTH; depth++) {
                System.out.printf("=== Expanding depth %d (frontier size: %d) ===%n", depth, frontier.size());

                int totalTerms = 0;

                long nextFrontierSize = frontier.size() * (EXPANSION_FACTOR - (depth - 1));
                Path nextFrontierPath = Paths.get("src/main/resources/MiniMax/next_frontier_depth" + depth + ".bin");
                FileChannel nextFC = new RandomAccessFile(new File(nextFrontierPath.toString()), "rw").getChannel();
                LongBuffer nextLB = nextFC.map(FileChannel.MapMode.READ_WRITE, 0, nextFrontierSize * Long.BYTES).asLongBuffer();

                for (int batchStart = 0; batchStart < frontier.size(); batchStart += MAX_BOARDS_PER_BATCH) {
                    int batchEnd = Math.min(batchStart + MAX_BOARDS_PER_BATCH, frontier.size());
                    int batchLen = batchEnd - batchStart;
                    LongArrayList batch = new LongArrayList(batchLen);
                    batch.add(frontier.buffer, batchStart, batchEnd);

                    // Expand & classify this batch
                    ExpandAndClassify.Result res = expander.run(batch, depth);

                    // Handle terminals as before (stream to disk)
                    int batchTermCount = res.termX.size() + res.termO.size() + res.termTie.size();
                    LongArrayList allBatchTerms = new LongArrayList(batchTermCount);
                    int idx = 0;

                    for (LongCursor b : res.termX)   allBatchTerms.set(idx++,b.value);
                    for (LongCursor b : res.termO)   allBatchTerms.set(idx++,b.value);
                    for (LongCursor b : res.termTie) allBatchTerms.set(idx++,b.value);
                    if (batchTermCount > 0) {
                        lb.put(allBatchTerms.buffer);
                    }
                    totalTerms += batchTermCount;

                    // Write non-terminals
                    for (LongCursor c : res.frontierChunks) {
                        LongArrayList chunk = LongArrayList.from(c.value);
                        for (LongCursor lc : chunk) {
                            nextLB.put(lc.value);
                        }
                    }

                }
                System.out.printf("Depth %d: %d terminal boards%n", depth, totalTerms);

                nextFC.close();

                // Now, for the next loop, read the file back into batches for processing
                LongArrayList nextFrontierList = new LongArrayList();
                FileChannel readFC = new FileInputStream(nextFrontierPath.toFile()).getChannel();
                LongBuffer readLB = readFC.map(FileChannel.MapMode.READ_ONLY, 0, readFC.size()).asLongBuffer();
                while (readLB.hasRemaining()) {
                    nextFrontierList.add(readLB.get());
                }
                frontier = nextFrontierList;
            }

            fc.close();

            System.out.println("Done! All terminal positions written to " + outFile);
        } catch (IOException e) {
            e.printStackTrace();
            System.err.println("Failed during expansion or writing.");
        }

    }
}
