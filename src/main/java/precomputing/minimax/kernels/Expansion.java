package precomputing.minimax.kernels;

import precomputing.minimax.*;
import org.jocl.*;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

import static org.jocl.CL.*;

public class Expansion {
    private static final int MAX_DEPTH = 27;
    private static final int CHILD_LEN = MAX_DEPTH + 1;

    private final cl_kernel detectKernel;
    private final cl_kernel expandKernel;

    public Expansion() {
        // build and extract kernels
        cl_program detProg = CLContextManager.buildProgram("TurnDetection.cl");
        detectKernel = clCreateKernel(detProg, "detectTurn", null);

        cl_program expProg = CLContextManager.buildProgram("ExpansionAll.cl");
        expandKernel = clCreateKernel(expProg, "expandAll", null);
    }

    /**
     * Expand each input board into exactly 27 children (no checking for used
     * positions—duplicates will be removed later). Uses two GPU kernels:
     * 1) detectTurn → produce a flag per board
     * 2) expandAll  → append all 27 possible moves
     */
    public List<String> expandAll(List<String> boards) {
        int n = boards.size();
        int inSize = n * MAX_DEPTH;
        int flagsSize = n;
        int outSize = n * 27 * CHILD_LEN;

        // 1) pack input boards into a contiguous byte array
        byte[] inBuf = new byte[inSize];
        for (int i = 0; i < n; i++) {
            byte[] b = boards.get(i).getBytes(StandardCharsets.US_ASCII);
            System.arraycopy(b, 0, inBuf, i * MAX_DEPTH, b.length);
        }

        // 2) allocate GPU buffers
        cl_mem memIn = clCreateBuffer(CLContextManager.getContext(),
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                inSize, Pointer.to(inBuf), null);

        cl_mem memFlags = clCreateBuffer(CLContextManager.getContext(),
                CL_MEM_READ_WRITE,
                flagsSize, null, null);

        // 3) run detectTurn
        clSetKernelArg(detectKernel, 0, Sizeof.cl_mem, Pointer.to(memIn));
        clSetKernelArg(detectKernel, 1, Sizeof.cl_mem, Pointer.to(memFlags));
        clEnqueueNDRangeKernel(
                CLContextManager.getQueue(),
                detectKernel,
                1, null,
                new long[]{n},
                null, 0, null, null
        );

        // 4) run expandAll
        cl_mem memOut = clCreateBuffer(CLContextManager.getContext(),
                CL_MEM_WRITE_ONLY,
                outSize, null, null);

        clSetKernelArg(expandKernel, 0, Sizeof.cl_mem, Pointer.to(memIn));
        clSetKernelArg(expandKernel, 1, Sizeof.cl_mem, Pointer.to(memFlags));
        clSetKernelArg(expandKernel, 2, Sizeof.cl_mem, Pointer.to(memOut));
        clEnqueueNDRangeKernel(
                CLContextManager.getQueue(),
                expandKernel,
                1, null,
                new long[]{n},
                null, 0, null, null
        );

        // 5) read back results
        byte[] outBuf = new byte[outSize];
        clEnqueueReadBuffer(
                CLContextManager.getQueue(),
                memOut, CL_TRUE,
                0, outSize,
                Pointer.to(outBuf),
                0, null, null
        );

        // 6) release GPU resources
        clReleaseMemObject(memIn);
        clReleaseMemObject(memFlags);
        clReleaseMemObject(memOut);

        // 7) unpack into Java Strings
        List<String> result = new ArrayList<>(n * 27);
        for (int i = 0; i < n * 27; i++) {
            int base = i * CHILD_LEN;
            int len = 0;
            while (len < CHILD_LEN && outBuf[base + len] != 0) {
                len++;
            }
            result.add(new String(outBuf, base, len, StandardCharsets.US_ASCII));
        }
        return result;
    }
}