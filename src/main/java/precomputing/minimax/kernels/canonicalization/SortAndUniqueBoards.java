package precomputing.minimax.kernels.canonicalization;

import support.*;
import org.jocl.*;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

import static org.jocl.CL.*;

public class SortAndUniqueBoards {
    private final cl_kernel kernel;
    private static final long MAX_BUFFER_BYTES = 100L * 1024 * 1024; // 100 MB

    public SortAndUniqueBoards() throws IOException {
        CL.setExceptionsEnabled(true);
        cl_program p = CLContextManager.buildProgram("SortAndUniqueBoards.cl");
        kernel = clCreateKernel(p, "sortBoard", null);
    }

    /**
     * Instead of returning a List<String>, we write the final sorted+unique boards
     * to `outputFile` on disk.  This will never OOM, no matter how many boards.
     */
    public Path sortAndUniqueToFile(List<String> boards) throws IOException {
        int N = boards.size();
        if (N == 0) {
            Path empty = Files.createTempFile("uniq_boards", ".txt");
            Files.deleteIfExists(empty);
            return empty;
        }

        // 1) Find max length
        int maxLen = boards.stream()
                .mapToInt(String::length)
                .max()
                .orElse(0);

        // 2) Compute batchSize so batchSize*maxLen ≤ MAX_BUFFER_BYTES
        int batchSize = (int) Math.max(1,
                Math.min(N, MAX_BUFFER_BYTES / (long) maxLen));

        List<Path> runs = new ArrayList<>();
        cl_context ctx = CLContextManager.getContext();
        cl_command_queue queue = CLContextManager.getQueue();

        // 3) For each batch: GPU-sort-&-unique, then write to a temp run file
        for (int start = 0; start < N; start += batchSize) {
            int end = Math.min(N, start + batchSize);
            int B = end - start;

            // 3a) pack input
            byte[] inBuf = new byte[B * maxLen];
            for (int i = 0; i < B; i++) {
                byte[] ascii = boards.get(start + i)
                        .getBytes(StandardCharsets.US_ASCII);
                System.arraycopy(ascii, 0,
                        inBuf, i * maxLen,
                        ascii.length);
            }

            // 3b) GPU buffers
            cl_mem memIn = clCreateBuffer(ctx,
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    inBuf.length, Pointer.to(inBuf), null);
            cl_mem memOut = clCreateBuffer(ctx,
                    CL_MEM_WRITE_ONLY,
                    inBuf.length, null, null);

            clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(memIn));
            clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(memOut));
            clSetKernelArg(kernel, 2, Sizeof.cl_int, Pointer.to(new int[]{maxLen}));

            clEnqueueNDRangeKernel(queue, kernel, 1, null,
                    new long[]{B}, null, 0, null, null);
            clEnqueueReadBuffer(queue, memOut, CL_TRUE,
                    0, inBuf.length, Pointer.to(inBuf),
                    0, null, null);

            // 3c) cleanup GPU
            clReleaseMemObject(memIn);
            clReleaseMemObject(memOut);

            // 3d) decode + batch-unique in-RAM
            List<String> lines = new ArrayList<>(B);
            for (int i = 0; i < B; i++) {
                int base = i * maxLen, len = 0;
                while (len < maxLen && inBuf[base + len] != 0) len++;
                lines.add(new String(inBuf, base, len,
                        StandardCharsets.US_ASCII));
            }
            // sort & in-batch unique
            Collections.sort(lines);
            List<String> uniqBatch = new ArrayList<>(lines.size());
            String last = null;
            for (String s : lines) {
                if (!s.equals(last)) {
                    uniqBatch.add(s);
                    last = s;
                }
            }
            lines = null;

            // 3e) write run file
            Path runFile = Files.createTempFile("uniq_run_", ".txt");
            try (BufferedWriter w = Files.newBufferedWriter(runFile)) {
                for (String s : uniqBatch) {
                    w.write(s);
                    w.newLine();
                }
            }
            runs.add(runFile);
        }

        // 4) Merge-sort all run-files into ONE final output
        Path output = Files.createTempFile("uniq_final_", ".txt");
        PriorityQueue<RunEntry> pq = new PriorityQueue<>();
        List<BufferedReader> readers = new ArrayList<>();
        try (BufferedWriter w = Files.newBufferedWriter(output)) {
            // open readers
            for (Path run : runs) {
                BufferedReader r = Files.newBufferedReader(run);
                readers.add(r);
                String line = r.readLine();
                if (line != null) pq.add(new RunEntry(line, r));
            }
            // k-way merge
            String last = null;
            while (!pq.isEmpty()) {
                RunEntry e = pq.poll();
                if (!e.line.equals(last)) {
                    w.write(e.line);
                    w.newLine();
                    last = e.line;
                }
                String next = e.reader.readLine();
                if (next != null) {
                    pq.add(new RunEntry(next, e.reader));
                }
            }
        } finally {
            // cleanup readers & temp runs
            for (BufferedReader r : readers) r.close();
            for (Path run : runs) Files.deleteIfExists(run);
        }

        return output;
    }

    public void release() {
        clReleaseKernel(kernel);
    }

    // helper for merging
    private static class RunEntry implements Comparable<RunEntry> {
        final String line;
        final BufferedReader reader;

        RunEntry(String line, BufferedReader reader) {
            this.line = line;
            this.reader = reader;
        }

        @Override
        public int compareTo(RunEntry o) {
            return this.line.compareTo(o.line);
        }
    }
}