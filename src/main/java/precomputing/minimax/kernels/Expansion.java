package precomputing.minimax.kernels;

import support.*;
import org.jocl.*;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

import static org.jocl.CL.*;

public class Expansion {
    private static final int MAX_DEPTH = 27;
    private static final int CHILD_LEN = MAX_DEPTH + 1;

    // Tune this to fit comfortably in your GPU memory (in boards count)
    private static final int CHUNK_SIZE = 10_000_000;

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
     * Expand all boards, but in manageable chunks to avoid OOM on the GPU.
     */
    public List<String> expandAll(List<String> boards) {
        List<String> allChildren = new ArrayList<>(boards.size() * 27);
        int total = boards.size();

        for (int start = 0; start < total; start += CHUNK_SIZE) {
            int end = Math.min(total, start + CHUNK_SIZE);
            List<String> slice = boards.subList(start, end);
            allChildren.addAll(expandChunk(slice));
        }

        return allChildren;
    }

    /**
     * Actually runs the two‐kernel expansion on a sublist of boards.
     */
    private List<String> expandChunk(List<String> boards) {
        int n = boards.size();
        int inSize = n * MAX_DEPTH;
        int outSize = n * 27 * CHILD_LEN;

        // 1) pack input boards into a contiguous byte array
        byte[] inBuf = new byte[inSize];
        for (int i = 0; i < n; i++) {
            byte[] b = boards.get(i).getBytes(StandardCharsets.US_ASCII);
            System.arraycopy(b, 0, inBuf, i * MAX_DEPTH, b.length);
        }

        // 2) allocate GPU buffers
        cl_mem memIn = clCreateBuffer(
                CLContextManager.getContext(),
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                inSize, Pointer.to(inBuf), null
        );
        cl_mem memFlags = clCreateBuffer(
                CLContextManager.getContext(),
                CL_MEM_READ_WRITE,
                n, null, null
        );
        cl_mem memOut = clCreateBuffer(
                CLContextManager.getContext(),
                CL_MEM_WRITE_ONLY,
                outSize, null, null
        );

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

        // 6) cleanup
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
