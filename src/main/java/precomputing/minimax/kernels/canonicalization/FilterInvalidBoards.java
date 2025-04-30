package precomputing.minimax.kernels.canonicalization;

import support.*;
import org.jocl.*;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

import static org.jocl.CL.*;

public class FilterInvalidBoards {
    private final cl_kernel kernel;
    private static final long MAX_BUFFER_BYTES = 100L * 1024 * 1024; // 100 MB

    public FilterInvalidBoards() {
        CL.setExceptionsEnabled(true);
        cl_program program = CLContextManager.buildProgram("FilterInvalidBoards.cl");
        kernel = clCreateKernel(program, "filterValid", null);
    }

    /**
     * @param boards list of move-strings
     * @return only those boards with no repeated positions
     */
    public List<String> filter(List<String> boards) {
        int N = boards.size();
        if (N == 0) return List.of();

        // max single‐board length
        int maxLen = boards.stream()
                .mapToInt(String::length)
                .max()
                .orElse(0);

        List<String> result = new ArrayList<>();
        cl_context ctx = CLContextManager.getContext();
        cl_command_queue q = CLContextManager.getQueue();

        // batch so batchSize*maxLen ≤ MAX_BUFFER_BYTES
        int batchSize = (int) Math.max(1, Math.min(N, MAX_BUFFER_BYTES / (long) maxLen));

        for (int start = 0; start < N; start += batchSize) {
            int end = Math.min(N, start + batchSize), B = end - start;

            // pack this chunk
            byte[] inBuf = new byte[B * maxLen];
            for (int i = 0; i < B; i++) {
                byte[] ascii = boards.get(start + i).getBytes(StandardCharsets.US_ASCII);
                System.arraycopy(ascii, 0, inBuf, i * maxLen, ascii.length);
            }

            // GPU buffers
            cl_mem memIn = clCreateBuffer(ctx,
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    inBuf.length, Pointer.to(inBuf), null);
            cl_mem memFlags = clCreateBuffer(ctx,
                    CL_MEM_WRITE_ONLY,
                    B * Sizeof.cl_int, null, null);

            // launch filterValid on B items
            clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(memIn));
            clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(memFlags));
            clSetKernelArg(kernel, 2, Sizeof.cl_int, Pointer.to(new int[]{maxLen}));
            clEnqueueNDRangeKernel(q, kernel, 1, null,
                    new long[]{B}, null, 0, null, null);

            // read back flags
            int[] flags = new int[B];
            clEnqueueReadBuffer(q, memFlags, CL_TRUE,
                    0, Sizeof.cl_int * B, Pointer.to(flags),
                    0, null, null);

            // cleanup
            clReleaseMemObject(memIn);
            clReleaseMemObject(memFlags);

            // collect survivors
            for (int i = 0; i < B; i++) {
                if (flags[i] == 1) {
                    result.add(boards.get(start + i));
                }
            }
        }

        return result;
    }

    public void release() {
        clReleaseKernel(kernel);
    }
}