package precomputing.minimax.kernels;

import support.*;
import org.jocl.*;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

import static org.jocl.CL.*;

public class RotationCanonicalizerGPU {
    private static final int MAX_DEPTH = 27;
    private final cl_kernel kernel;

    public RotationCanonicalizerGPU() {
        cl_program p = CLContextManager.buildProgram("RotationCanonicalization.cl");
        kernel = clCreateKernel(p, "rotateCanon", null);
    }

    /**
     * @param boards list of zero-padded strings (length ≤ MAX_DEPTH)
     * @param step   current board length
     * @return canonicalized strings (zero-padded)
     */
    public List<String> canonicalize(List<String> boards, int step) {
        int n = boards.size();
        int bufSize = n * MAX_DEPTH;

        // pack input
        byte[] inBuf = new byte[bufSize];
        for (int i = 0; i < n; i++) {
            byte[] ascii = boards.get(i).getBytes(StandardCharsets.US_ASCII);
            System.arraycopy(ascii, 0, inBuf, i * MAX_DEPTH, ascii.length);
        }

        // alloc GPU buffers
        cl_mem memIn = clCreateBuffer(
                CLContextManager.getContext(),
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                bufSize, Pointer.to(inBuf), null);
        cl_mem memOut = clCreateBuffer(
                CLContextManager.getContext(),
                CL_MEM_WRITE_ONLY,
                bufSize, null, null);

        // set args & launch
        clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(memIn));
        clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(memOut));
        clSetKernelArg(kernel, 2, Sizeof.cl_int, Pointer.to(new int[]{step}));

        int localSize = 256;
        int globalSize = ((n + localSize - 1) / localSize) * localSize;

        clEnqueueNDRangeKernel(CLContextManager.getQueue(), kernel,
                1, null,
                new long[]{globalSize},
                new long[]{localSize},
                0, null, null);

        // read back
        byte[] outBuf = new byte[bufSize];
        clEnqueueReadBuffer(
                CLContextManager.getQueue(),
                memOut, CL_TRUE,
                0, bufSize,
                Pointer.to(outBuf),
                0, null, null);

        clReleaseMemObject(memIn);
        clReleaseMemObject(memOut);

        // unpack to Java Strings
        List<String> result = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            int off = i * MAX_DEPTH;
            int len = 0;
            while (len < MAX_DEPTH && outBuf[off + len] != 0) {
                len++;
            }
            result.add(new String(outBuf, off, len, StandardCharsets.US_ASCII));
        }
        return result;
    }
}