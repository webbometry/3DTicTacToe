// RotationKernel.java

package precomputing.minimax.kernels.canonicalization;

import support.CLContextManager;
import org.jocl.*;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

import static org.jocl.CL.*;

public class RotationKernel {
    private static final int MAP_COUNT = 24;
    private static final int MAP_SIZE = 27;
    private final cl_kernel kernel;
    private final cl_mem memMaps;
    private static final long MAX_BUFFER_BYTES = 100L * 1024 * 1024; // 100 MB

    public RotationKernel(Path rotationMapsPath) throws IOException {
        CL.setExceptionsEnabled(true);
        cl_program program = CLContextManager.buildProgram("RotationKernel.cl");
        kernel = clCreateKernel(program, "collapseRotation", null);

        List<String> lines = Files.readAllLines(rotationMapsPath, StandardCharsets.UTF_8);
        int[] maps = new int[MAP_COUNT * MAP_SIZE];
        for (int r = 0; r < MAP_COUNT; r++) {
            String[] parts = lines.get(r).trim().split("\\s+");
            for (int i = 0; i < MAP_SIZE; i++) {
                maps[r * MAP_SIZE + i] = Integer.parseInt(parts[i]);
            }
        }

        cl_context ctx = CLContextManager.getContext();
        memMaps = clCreateBuffer(ctx,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_int * maps.length,
                Pointer.to(maps),
                null);
    }

    /**
     * Remove rotational duplicates (keep first‐seen) in batches.
     */
    public long[] collapseRotation(long[] codes) {
        int N = codes.length;
        if (N <= 1) {
            return codes.clone();
        }

        // --- Phase A: GPU compute minimal rotation code per board (unchanged) ---
        cl_context ctx = CLContextManager.getContext();
        cl_command_queue queue = CLContextManager.getQueue();

        int batchSize = (int) Math.max(1,
                Math.min(N, MAX_BUFFER_BYTES / (2L * Sizeof.cl_ulong)));

        long[] minCodes = new long[N];
        for (int start = 0; start < N; start += batchSize) {
            int end = Math.min(N, start + batchSize), B = end - start;

            long[] inBatch = new long[B];
            System.arraycopy(codes, start, inBatch, 0, B);

            cl_mem memIn = clCreateBuffer(ctx,
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    Sizeof.cl_ulong * B, Pointer.to(inBatch), null);
            cl_mem memOut = clCreateBuffer(ctx,
                    CL_MEM_WRITE_ONLY,
                    Sizeof.cl_ulong * B, null, null);

            clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(memIn));
            clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(memMaps));
            clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(memOut));

            clEnqueueNDRangeKernel(queue, kernel, 1,
                    null, new long[]{B}, null, 0, null, null);

            long[] outBatch = new long[B];
            clEnqueueReadBuffer(queue, memOut, CL_TRUE,
                    0, Sizeof.cl_ulong * B,
                    Pointer.to(outBatch), 0, null, null);

            clReleaseMemObject(memIn);
            clReleaseMemObject(memOut);

            System.arraycopy(outBatch, 0, minCodes, start, B);
        }

        // --- Phase B: stable sort indices by minCodes[i] in parallel ---
        Integer[] idx = new Integer[N];
        for (int i = 0; i < N; i++) {
            idx[i] = i;
        }
        // stable, parallel mergesort under the hood
        Arrays.parallelSort(idx, Comparator.comparingLong(i -> minCodes[i]));

        // --- Phase C: scan sorted indices, emitting first-seen originals ---
        long[] temp = new long[N];
        int out = 0;
        long last = Long.MIN_VALUE;  // codes are non-negative 54-bit values
        for (int j = 0; j < N; j++) {
            int i = idx[j];
            long key = minCodes[i];
            if (j == 0 || key != last) {
                temp[out++] = codes[i];
                last = key;
            }
        }

        // trim and return
        return Arrays.copyOf(temp, out);
    }

    public void release() {
        clReleaseMemObject(memMaps);
        clReleaseKernel(kernel);
    }
}