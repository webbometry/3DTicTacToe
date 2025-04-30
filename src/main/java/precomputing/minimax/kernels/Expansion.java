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

    // chunkSize chosen so that outSize ≤ MAX_GPU_BYTES
    private final int chunkSize;
    private final cl_kernel detectKernel, expandKernel;

    public Expansion() {
        this.chunkSize = (int) (MAX_GPU_BYTES / (27L * CHILD_LEN));
        if (chunkSize < 1) throw new IllegalStateException("GPU buffer too small");

        cl_program detProg = CLContextManager.buildProgram("TurnDetection.cl");
        detectKernel = clCreateKernel(detProg, "detectTurn", null);
        cl_program expProg = CLContextManager.buildProgram("ExpansionAll.cl");
        expandKernel = clCreateKernel(expProg, "expandAll", null);
    }

    public List<String> expandAll(List<String> boards) {
        int N = boards.size();
        List<String> allChildren = new ArrayList<>();

        // only ever feed at most `chunkSize` boards per GPU call
        for (int start = 0; start < N; start += chunkSize) {
            int end = Math.min(N, start + chunkSize);
            allChildren.addAll(expandChunk(boards.subList(start, end)));
        }

        return allChildren;
    }

    private List<String> expandChunk(List<String> boards) {
        int n = boards.size();
        int inSize = n * MAX_DEPTH;
        int outSize = n * 27 * CHILD_LEN;

        byte[] inBuf = new byte[inSize];
        byte[] outBuf = new byte[outSize];

        for (int i = 0; i < n; i++) {
            byte[] b = boards.get(i).getBytes(StandardCharsets.US_ASCII);
            System.arraycopy(b, 0, inBuf, i * MAX_DEPTH, b.length);
        }

        cl_context ctx = CLContextManager.getContext();
        cl_command_queue queue = CLContextManager.getQueue();

        cl_mem memIn = clCreateBuffer(ctx,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, inSize, Pointer.to(inBuf), null);
        cl_mem memFlag = clCreateBuffer(ctx,
                CL_MEM_READ_WRITE, n, null, null);
        cl_mem memOut = clCreateBuffer(ctx,
                CL_MEM_WRITE_ONLY, outSize, null, null);

        // detect turn
        clSetKernelArg(detectKernel, 0, Sizeof.cl_mem, Pointer.to(memIn));
        clSetKernelArg(detectKernel, 1, Sizeof.cl_mem, Pointer.to(memFlag));
        clEnqueueNDRangeKernel(queue, detectKernel, 1, null, new long[]{n}, null, 0, null, null);

        // expand
        clSetKernelArg(expandKernel, 0, Sizeof.cl_mem, Pointer.to(memIn));
        clSetKernelArg(expandKernel, 1, Sizeof.cl_mem, Pointer.to(memFlag));
        clSetKernelArg(expandKernel, 2, Sizeof.cl_mem, Pointer.to(memOut));
        clEnqueueNDRangeKernel(queue, expandKernel, 1, null, new long[]{n}, null, 0, null, null);

        // read back
        clEnqueueReadBuffer(queue, memOut, CL_TRUE, 0, outSize, Pointer.to(outBuf), 0, null, null);

        clFinish(queue);
        clReleaseMemObject(memIn);
        clReleaseMemObject(memFlag);
        clReleaseMemObject(memOut);

        // unpack into Strings
        List<String> result = new ArrayList<>(n * 27);
        for (int i = 0; i < n * 27; i++) {
            int base = i * CHILD_LEN, len = 0;
            while (len < CHILD_LEN && outBuf[base + len] != 0) len++;
            result.add(new String(outBuf, base, len, StandardCharsets.US_ASCII));
        }
        return result;
    }
}