package precomputing.minimax;

import support.CLContext;
import org.jocl.*;

import java.util.ArrayList;
import java.util.List;

import static org.jocl.CL.*;

/**
 * Wraps the expand_and_classify kernel, but returns the frontier
 * in chunks of size ≤ maxBoards to avoid giant arrays.
 */
public class ExpandAndClassify {
    private final CLContext cl;
    private final int       maxBoards;  // maximum longs per chunk

    public ExpandAndClassify(CLContext cl, int maxBoards) {
        this.cl        = cl;
        this.maxBoards = maxBoards;
    }

    /** Result holds many small frontier chunks plus the terminals. */
    public static class Result {
        public final List<long[]> frontierChunks;
        public final long[] termX, termO;
        public Result(List<long[]> fc, long[] x, long[] o) {
            this.frontierChunks = fc;
            this.termX = x;
            this.termO = o;
        }
    }

    public Result run(long[] inputBoards, int depth) {
        int inCount   = inputBoards.length;
        int totalThds = inCount * 27;
        int maxOut    = inCount * 27;

        // 1) Allocate device buffers
        cl_mem bufIn = clCreateBuffer(cl.ctx,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_ulong * inCount,
                Pointer.to(inputBoards), null);

        cl_mem bufFr = clCreateBuffer(cl.ctx,
                CL_MEM_READ_WRITE,
                Sizeof.cl_ulong * maxOut, null, null);

        cl_mem bufTX = clCreateBuffer(cl.ctx,
                CL_MEM_READ_WRITE,
                Sizeof.cl_ulong * maxOut, null, null);

        cl_mem bufTO = clCreateBuffer(cl.ctx,
                CL_MEM_READ_WRITE,
                Sizeof.cl_ulong * maxOut, null, null);

        // counters
        cl_mem ctrF = clCreateBuffer(cl.ctx, CL_MEM_READ_WRITE, Sizeof.cl_ulong, null, null);
        cl_mem ctrX = clCreateBuffer(cl.ctx, CL_MEM_READ_WRITE, Sizeof.cl_ulong, null, null);
        cl_mem ctrO = clCreateBuffer(cl.ctx, CL_MEM_READ_WRITE, Sizeof.cl_ulong, null, null);

        // zero all counters
        clEnqueueFillBuffer(cl.queue, ctrF, Pointer.to(new long[]{0L}),
                Sizeof.cl_ulong, 0, Sizeof.cl_ulong, 0, null, null);
        clEnqueueFillBuffer(cl.queue, ctrX, Pointer.to(new long[]{0L}),
                Sizeof.cl_ulong, 0, Sizeof.cl_ulong, 0, null, null);
        clEnqueueFillBuffer(cl.queue, ctrO, Pointer.to(new long[]{0L}),
                Sizeof.cl_ulong, 0, Sizeof.cl_ulong, 0, null, null);

        // 2) Set kernel args
        cl_kernel kernel = cl.kernel;
        int ai = 0;
        clSetKernelArg(kernel, ai++, Sizeof.cl_mem, Pointer.to(bufIn));
        clSetKernelArg(kernel, ai++, Sizeof.cl_uint, Pointer.to(new int[]{inCount}));
        clSetKernelArg(kernel, ai++, Sizeof.cl_uint, Pointer.to(new int[]{depth}));
        clSetKernelArg(kernel, ai++, Sizeof.cl_mem, Pointer.to(bufFr));
        clSetKernelArg(kernel, ai++, Sizeof.cl_mem, Pointer.to(ctrF));
        clSetKernelArg(kernel, ai++, Sizeof.cl_mem, Pointer.to(bufTX));
        clSetKernelArg(kernel, ai++, Sizeof.cl_mem, Pointer.to(ctrX));
        clSetKernelArg(kernel, ai++, Sizeof.cl_mem, Pointer.to(bufTO));
        clSetKernelArg(kernel, ai++, Sizeof.cl_mem, Pointer.to(ctrO));

        // 3) Launch
        long localSize  = 128;
        long globalSize = ((totalThds + localSize - 1) / localSize) * localSize;
        clEnqueueNDRangeKernel(cl.queue, kernel, 1, null,
                new long[]{globalSize}, new long[]{localSize},
                0, null, null);

        // 4) Read back terminal counts
        long[] cntX = new long[1], cntO = new long[1], cntF = new long[1];
        clEnqueueReadBuffer(cl.queue, ctrX, CL_TRUE, 0, Sizeof.cl_ulong, Pointer.to(cntX), 0, null, null);
        clEnqueueReadBuffer(cl.queue, ctrO, CL_TRUE, 0, Sizeof.cl_ulong, Pointer.to(cntO), 0, null, null);
        clEnqueueReadBuffer(cl.queue, ctrF, CL_TRUE, 0, Sizeof.cl_ulong, Pointer.to(cntF), 0, null, null);

        // 5) Read back terminal boards (OK to be large; typically << frontier size)
        long[] termX = new long[(int)cntX[0]];
        long[] termO = new long[(int)cntO[0]];
        if (termX.length > 0)
            clEnqueueReadBuffer(cl.queue, bufTX, CL_TRUE, 0,
                    termX.length * Sizeof.cl_ulong, Pointer.to(termX), 0, null, null);
        if (termO.length > 0)
            clEnqueueReadBuffer(cl.queue, bufTO, CL_TRUE, 0,
                    termO.length * Sizeof.cl_ulong, Pointer.to(termO), 0, null, null);

        // 6) Read back the frontier in chunks of ≤ maxBoards
        List<long[]> frontierChunks = new ArrayList<>();
        long remaining = cntF[0];
        long off       = 0;
        while (remaining > 0) {
            int chunkSize = (int)Math.min(maxBoards, remaining);
            long[] chunk = new long[chunkSize];
            clEnqueueReadBuffer(cl.queue, bufFr, CL_TRUE,
                    off * Sizeof.cl_ulong,
                    chunkSize * Sizeof.cl_ulong,
                    Pointer.to(chunk), 0, null, null);
            frontierChunks.add(chunk);
            off       += chunkSize;
            remaining -= chunkSize;
        }

        // 7) Release everything
        for (cl_mem m : new cl_mem[]{bufIn, bufFr, bufTX, bufTO, ctrF, ctrX, ctrO}) {
            clReleaseMemObject(m);
        }

        return new Result(frontierChunks, termX, termO);
    }
}
