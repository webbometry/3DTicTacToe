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
    private static final long MAX_GPU_BYTES = 512L * 1024 * 1024;

    private final int chunkSize;
    private final cl_kernel detectKernel;
    private final cl_kernel expandKernel;

    public Expansion() {
        // figure out how many boards keeps outSize ≤ MAX_GPU_BYTES
        this.chunkSize = (int) (MAX_GPU_BYTES / (27L * CHILD_LEN));
        if (chunkSize < 1) {
            throw new IllegalStateException("MAX_GPU_BYTES too small for even one board");
        }

        // compile kernels
        cl_program detProg = CLContextManager.buildProgram("TurnDetection.cl");
        detectKernel = clCreateKernel(detProg, "detectTurn", null);
        cl_program expProg = CLContextManager.buildProgram("ExpansionAll.cl");
        expandKernel = clCreateKernel(expProg, "expandAll", null);
    }

    public List<String> expandAll(List<String> boards) {
        List<String> allChildren = new ArrayList<>(boards.size() * 27);
        for (int start = 0; start < boards.size(); start += chunkSize) {
            int end = Math.min(boards.size(), start + chunkSize);
            allChildren.addAll(expandChunk(boards.subList(start, end)));
        }
        return allChildren;
    }

    private List<String> expandChunk(List<String> boards) {
        int n = boards.size();
        int inSize = n * MAX_DEPTH;
        int outSize = n * 27 * CHILD_LEN; // ≤ MAX_GPU_BYTES guaranteed

        byte[] inBuf = new byte[inSize];
        for (int i = 0; i < n; i++) {
            byte[] b = boards.get(i).getBytes(StandardCharsets.US_ASCII);
            System.arraycopy(b, 0, inBuf, i * MAX_DEPTH, b.length);
        }

        cl_context ctx = CLContextManager.getContext();
        cl_command_queue q = CLContextManager.getQueue();

        cl_mem memIn = clCreateBuffer(ctx,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                inSize, Pointer.to(inBuf), null);
        cl_mem memFlags = clCreateBuffer(ctx,
                CL_MEM_READ_WRITE,
                n, null, null);
        cl_mem memOut = clCreateBuffer(ctx,
                CL_MEM_WRITE_ONLY,
                outSize, null, null);

        // detect turn
        clSetKernelArg(detectKernel, 0, Sizeof.cl_mem, Pointer.to(memIn));
        clSetKernelArg(detectKernel, 1, Sizeof.cl_mem, Pointer.to(memFlags));
        clEnqueueNDRangeKernel(q, detectKernel, 1, null, new long[]{n}, null, 0, null, null);

        // expand
        clSetKernelArg(expandKernel, 0, Sizeof.cl_mem, Pointer.to(memIn));
        clSetKernelArg(expandKernel, 1, Sizeof.cl_mem, Pointer.to(memFlags));
        clSetKernelArg(expandKernel, 2, Sizeof.cl_mem, Pointer.to(memOut));
        clEnqueueNDRangeKernel(q, expandKernel, 1, null, new long[]{n}, null, 0, null, null);

        // read back
        byte[] outBuf = new byte[outSize];
        clEnqueueReadBuffer(q, memOut, CL_TRUE, 0, outSize, Pointer.to(outBuf), 0, null, null);

        // clean up & ensure nothing is left queued
        clFinish(q);
        clReleaseMemObject(memIn);
        clReleaseMemObject(memFlags);
        clReleaseMemObject(memOut);

        // unpack results
        List<String> result = new ArrayList<>(n * 27);
        for (int i = 0; i < n * 27; i++) {
            int base = i * CHILD_LEN, len = 0;
            while (len < CHILD_LEN && outBuf[base + len] != 0) len++;
            result.add(new String(outBuf, base, len, StandardCharsets.US_ASCII));
        }
        return result;
    }
}