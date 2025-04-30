package precomputing.minimax.kernels.canonicalization;

import support.*;
import org.jocl.*;

import java.nio.charset.StandardCharsets;
import java.util.List;

import static org.jocl.CL.*;

public class SignatureKernel {
    private static final int MAX_DEPTH = 27;
    private final cl_kernel kernel;
    private static final long MAX_BUFFER_BYTES = 100L * 1024 * 1024; // 100 MB

    public SignatureKernel() {
        CL.setExceptionsEnabled(true);
        cl_program p = CLContextManager.buildProgram("SignatureKernel.cl");
        kernel = clCreateKernel(p, "buildSignature", null);
    }

    /**
     * @param boards zero‐padded strings (each ≤ MAX_DEPTH)
     * @param step   current move count
     * @return array of N packed 54‐bit codes
     */
    public long[] computeSignatures(List<String> boards, int step) {
        if (boards == null || boards.isEmpty()) {
            return new long[0];
        }

        int N = boards.size();
        long[] codes = new long[N];
        cl_context ctx = CLContextManager.getContext();
        cl_command_queue q = CLContextManager.getQueue();

        // batch so batchSize*MAX_DEPTH ≤ MAX_BUFFER_BYTES
        int batchSize = (int) Math.max(1, Math.min(N, MAX_BUFFER_BYTES / MAX_DEPTH));

        for (int start = 0; start < N; start += batchSize) {
            int end = Math.min(N, start + batchSize), B = end - start;

            // pack chunk
            byte[] inBuf = new byte[B * MAX_DEPTH];
            for (int i = 0; i < B; i++) {
                byte[] ascii = boards.get(start + i).getBytes(StandardCharsets.US_ASCII);
                System.arraycopy(ascii, 0, inBuf, i * MAX_DEPTH, ascii.length);
            }

            // GPU buffers
            cl_mem memIn = clCreateBuffer(ctx,
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    inBuf.length, Pointer.to(inBuf), null);
            cl_mem memOut = clCreateBuffer(ctx,
                    CL_MEM_WRITE_ONLY,
                    B * Sizeof.cl_ulong, null, null);

            // launch
            clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(memIn));
            clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(memOut));
            clSetKernelArg(kernel, 2, Sizeof.cl_int, Pointer.to(new int[]{step}));
            clEnqueueNDRangeKernel(q, kernel, 1, null, new long[]{B}, null, 0, null, null);

            // read back
            long[] batchCodes = new long[B];
            clEnqueueReadBuffer(q, memOut, CL_TRUE,
                    0, B * Sizeof.cl_ulong, Pointer.to(batchCodes), 0, null, null);

            // cleanup
            clReleaseMemObject(memIn);
            clReleaseMemObject(memOut);

            System.arraycopy(batchCodes, 0, codes, start, B);
        }

        return codes;
    }
}