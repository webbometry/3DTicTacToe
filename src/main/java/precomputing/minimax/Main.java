package precomputing.minimax;

import support.CLContext;
import support.DiskWriter;

import java.io.*;
import java.nio.file.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

public class Main {
    public static void main(String[] args) throws Exception {
        // 1) Init OpenCL
        CLContext cl = new CLContext("cl/expand_and_classify.cl");

        // 2) Compute max batch size
        long perBoardBytes =
                8L            // input
                        + 27L*8L        // frontier out
                        + 27L*8L        // termX out
                        + 27L*8L        // termO out
                        + 3L*8L;        // counters
        int maxBoards = (int)(cl.maxAllocBytes / perBoardBytes);
        System.out.printf("Using batch size = %,d boards%n", maxBoards);

        // 3) Setup GPU wrapper & terminal writer
        ExpandAndClassify exp = new ExpandAndClassify(cl, maxBoards);
        DiskWriter       disk = new DiskWriter(Paths.get("terminals.bin"));

        // 4) Heartbeat every minute
        ScheduledExecutorService hb = Executors.newSingleThreadScheduledExecutor();
        AtomicInteger hbCount = new AtomicInteger();
        DateTimeFormatter fmt = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
        hb.scheduleAtFixedRate(() -> {
            String now = LocalDateTime.now().format(fmt);
            System.out.printf("[%s] Still running #%d%n", now, hbCount.incrementAndGet());
        }, 1, 1, TimeUnit.MINUTES);

        // 5) BFS—disk‐backed
        int depth = 0;
        long totalBoards = 0, totalTerms = 0;
        long start = System.nanoTime();

        // Initialize first frontier file with the single empty board
        Path frontierFile = Files.createTempFile("frontier", ".bin");
        try (DataOutputStream w = new DataOutputStream(
                new BufferedOutputStream(Files.newOutputStream(frontierFile)))) {
            w.writeLong(0L);
        }

        while (true) {
            depth++;
            System.out.printf("=== Depth %d ===%n", depth);

            // Prepare next‐frontier file
            Path nextFile = Files.createTempFile("frontier", ".bin");
            try (DataInputStream  in  = new DataInputStream(
                    new BufferedInputStream(Files.newInputStream(frontierFile)));
                 DataOutputStream out = new DataOutputStream(
                         new BufferedOutputStream(Files.newOutputStream(nextFile)))
            ) {
                // Read in batches
                long[] batch = new long[maxBoards];
                while (true) {
                    int i = 0;
                    try {
                        for (; i < maxBoards; i++) {
                            batch[i] = in.readLong();
                        }
                    } catch (EOFException eof) { }
                    if (i == 0) break;  // no more boards

                    // Expand this batch
                    long[] toProcess = (i == maxBoards) ? batch : java.util.Arrays.copyOf(batch, i);
                    totalBoards += toProcess.length;
                    var res = exp.run(toProcess, depth);

                    // Write terminals
                    for (long b : res.termX) { disk.write(b, (byte)+27); totalTerms++; }
                    for (long b : res.termO) { disk.write(b, (byte)-27); totalTerms++; }

                    // Append next‐frontier boards
                    for (long nb : res.frontierChunks.stream().flatMapToLong(arr -> java.util.Arrays.stream(arr)).toArray()) {
                        out.writeLong(nb);
                    }
                }
            }

            // Delete old frontier, swap in new
            Files.delete(frontierFile);
            frontierFile = nextFile;

            // Quick check: is new frontier empty?
            if (Files.size(frontierFile) == 0) break;
        }

        // Shutdown
        hb.shutdownNow();
        disk.finish();
        double secs = (System.nanoTime() - start)/1e9;
        System.out.printf(
                "Expanded %,d boards, found %,d terminals in %.3f s%n",
                totalBoards, totalTerms, secs
        );
    }
}
