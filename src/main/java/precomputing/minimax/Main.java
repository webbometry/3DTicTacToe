package precomputing.minimax;

import support.CLContext;

import java.io.*;
import java.nio.file.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

public class Main {
    public static void main(String[] args) throws Exception {
        //----------------------------------------
        // 1) Directories on D:
        //----------------------------------------
        Path scratchRoot       = Paths.get("D:\\3dttt\\scratch");
        Path frontierDir       = scratchRoot.resolve("frontier");
        Path nextFrontierDir   = scratchRoot.resolve("frontier_next");
        Path termDir           = Paths.get("D:\\3dttt\\terminals");
        Files.createDirectories(frontierDir);
        Files.createDirectories(termDir);

        // Cleanup old frontier chunks
        System.out.println("Cleaning old frontier chunks...");
        if (Files.exists(frontierDir)) {
            Files.walk(frontierDir)
                    .sorted(Comparator.reverseOrder())
                    .forEach(p -> p.toFile().delete());
        }
        // Cleanup old terminal chunks
        System.out.println("Cleaning old terminal chunks...");
        if (Files.exists(termDir)) {
            Files.walk(termDir)
                    .sorted(Comparator.reverseOrder())
                    .forEach(p -> p.toFile().delete());
        }
        Files.createDirectories(frontierDir);
        Files.createDirectories(termDir);

        //----------------------------------------
        // 2) OpenCL setup
        //----------------------------------------
        CLContext cl = new CLContext("cl/expand_and_classify.cl");
        long perBoardBytes =  8L                // input
                + 27L*8L            // frontier out
                + 27L*8L            // termX out
                + 27L*8L            // termO out
                + 27L*8L            // termTie out
                + 3L*8L;            // 3 counters
        int maxBoards = (int)(cl.maxAllocBytes / perBoardBytes);
        System.out.printf("Using batch size = %,d boards%n", maxBoards);
        ExpandAndClassify exp = new ExpandAndClassify(cl, maxBoards);

        //----------------------------------------
        // 3) Heartbeat
        //----------------------------------------
        ScheduledExecutorService hb = Executors.newSingleThreadScheduledExecutor();
        AtomicInteger hbCount = new AtomicInteger();
        DateTimeFormatter dtf = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
        hb.scheduleAtFixedRate(() -> {
            System.out.printf("[%s] Still working… heartbeat #%d%n",
                    LocalDateTime.now().format(dtf), hbCount.incrementAndGet());
        }, 1, 1, TimeUnit.MINUTES);

        //----------------------------------------
        // 4) Seed depth 0
        //----------------------------------------
        int depth = 0;
        long totalBoards = 0, totalTerminals = 0;
        long startNano = System.nanoTime();

        // initial frontier file (one board: 0L)
        Path first = frontierDir.resolve("d00_fc000.bin");
        try (DataOutputStream w = new DataOutputStream(
                new BufferedOutputStream(Files.newOutputStream(first)))) {
            w.writeLong(0L);
        }

        //----------------------------------------
        // 5) BFS loop
        //----------------------------------------
        while (true) {
            depth++;
            System.out.println("=== Depth " + depth + " ===");

            // prepare next frontier dir
            if (Files.exists(nextFrontierDir))
                Files.walk(nextFrontierDir)
                        .sorted(Comparator.reverseOrder())
                        .forEach(p -> p.toFile().delete());
            Files.createDirectories(nextFrontierDir);

            // process each frontier chunk file
            List<Path> chunks = Files.list(frontierDir)
                    .filter(p -> p.toString().endsWith(".bin"))
                    .sorted()
                    .collect(Collectors.toList());

            int fcIdx = 0;
            for (Path fFile : chunks) {
                // load the entire chunk into memory (≤ maxBoards)
                long fileBytes = Files.size(fFile);
                int count = (int)(fileBytes / Long.BYTES);
                long[] batch = new long[count];
                try (DataInputStream in = new DataInputStream(
                        new BufferedInputStream(Files.newInputStream(fFile)))) {
                    for (int i = 0; i < count; i++) {
                        batch[i] = in.readLong();
                    }
                }

                totalBoards += count;
                var res = exp.run(batch, depth);

                // write terminals **only** from depth 9 onward
                if (depth >= 9) {
                    Path termChunk = termDir.resolve(
                            String.format("d%02d_fc%04d.bin", depth, fcIdx));
                    try (DataOutputStream tw = new DataOutputStream(
                            new BufferedOutputStream(Files.newOutputStream(termChunk)))) {
                        for (long b : res.termX)   { tw.writeLong(b); tw.writeByte(+27); totalTerminals++; }
                        for (long b : res.termO)   { tw.writeLong(b); tw.writeByte(-27); totalTerminals++; }
                        for (long b : res.termTie) { tw.writeLong(b); tw.writeByte(  0); totalTerminals++; }
                    }
                }

                // emit next-frontier chunks exactly as returned
                int n = 0;
                for (long[] fc : res.frontierChunks) {
                    Path outF = nextFrontierDir.resolve(
                            String.format("d%02d_fc%04d_%02d.bin", depth, fcIdx, n++));
                    try (DataOutputStream fw = new DataOutputStream(
                            new BufferedOutputStream(Files.newOutputStream(outF)))) {
                        for (long nb : fc) fw.writeLong(nb);
                    }
                }

                fcIdx++;
            }

            // swap dirs
            Files.walk(frontierDir)
                    .sorted(Comparator.reverseOrder())
                    .forEach(p -> p.toFile().delete());
            Files.createDirectories(frontierDir);
            Files.list(nextFrontierDir).forEach(p -> {
                try { Files.move(p, frontierDir.resolve(p.getFileName())); }
                catch(IOException e){ throw new UncheckedIOException(e); }
            });

            // if no files left, done
            if (Files.list(frontierDir).noneMatch(p -> p.toString().endsWith(".bin")))
                break;
        }

        //----------------------------------------
        // 6) Merge terminal chunks
        //----------------------------------------
        try (OutputStream finalOut = new BufferedOutputStream(
                Files.newOutputStream(Paths.get("D:\\3dttt\\terminals.bin")))) {
            Files.list(termDir)
                    .filter(p -> p.toString().endsWith(".bin"))
                    .sorted()
                    .forEach(p -> {
                        try (InputStream in = new BufferedInputStream(Files.newInputStream(p))) {
                            in.transferTo(finalOut);
                        } catch(IOException e){
                            throw new UncheckedIOException(e);
                        }
                    });
        }

        hb.shutdownNow();
        double secs = (System.nanoTime() - startNano) / 1e9;
        System.out.printf(
                "Done: processed %,d boards, wrote %,d terminals in %.3f s%n",
                totalBoards, totalTerminals, secs
        );
    }
}
