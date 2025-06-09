package precomputing.minimax;

import support.CLContext;
import org.jocl.*;

import java.util.ArrayList;
import java.util.List;

import static org.jocl.CL.*;

/**
 * GPU expansion + classification; returns frontier in safe chunks,
 * plus termX, termO, termTie arrays.
 */
public class ExpandAndClassify {
    private final CLContext cl;
    private final int       maxBoards;

    public static class Result {
        public final List<long[]> frontierChunks;
        public final long[] termX, termO, termTie;
        public Result(List<long[]> fc,
                      long[] x, long[] o, long[] t) {
            this.frontierChunks = fc;
            this.termX = x;
            this.termO = o;
            this.termTie = t;
        }
    }

    public ExpandAndClassify(CLContext cl, int maxBoards) {
        this.cl        = cl;
        this.maxBoards = maxBoards;
    }

    public Result run(long[] inputBoards, int depth) {
        int inCount = inputBoards.length;
        int totalThds = inCount * 27;
        int maxOut    = inCount * 27;

        // 1) Allocate device buffers
        cl_mem bufIn  = clCreateBuffer(cl.ctx,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_ulong * inCount,
                Pointer.to(inputBoards), null);

        cl_mem bufFr  = clCreateBuffer(cl.ctx,
                CL_MEM_READ_WRITE,
                Sizeof.cl_ulong * maxOut, null, null);

        cl_mem bufTX  = clCreateBuffer(cl.ctx,
                CL_MEM_READ_WRITE,
                Sizeof.cl_ulong * maxOut, null, null);

        cl_mem bufTO  = clCreateBuffer(cl.ctx,
                CL_MEM_READ_WRITE,
                Sizeof.cl_ulong * maxOut, null, null);

        cl_mem bufTT  = clCreateBuffer(cl.ctx,
                CL_MEM_READ_WRITE,
                Sizeof.cl_ulong * maxOut, null, null);

        // atomic counters
        cl_mem ctrF = clCreateBuffer(cl.ctx, CL_MEM_READ_WRITE, Sizeof.cl_ulong, null, null);
        cl_mem ctrX = clCreateBuffer(cl.ctx, CL_MEM_READ_WRITE, Sizeof.cl_ulong, null, null);
        cl_mem ctrO = clCreateBuffer(cl.ctx, CL_MEM_READ_WRITE, Sizeof.cl_ulong, null, null);
        cl_mem ctrT = clCreateBuffer(cl.ctx, CL_MEM_READ_WRITE, Sizeof.cl_ulong, null, null);

        // zero them
        clEnqueueFillBuffer(cl.queue, ctrF, Pointer.to(new long[]{0L}),
                Sizeof.cl_ulong, 0, Sizeof.cl_ulong, 0, null, null);
        clEnqueueFillBuffer(cl.queue, ctrX, Pointer.to(new long[]{0L}),
                Sizeof.cl_ulong, 0, Sizeof.cl_ulong, 0, null, null);
        clEnqueueFillBuffer(cl.queue, ctrO, Pointer.to(new long[]{0L}),
                Sizeof.cl_ulong, 0, Sizeof.cl_ulong, 0, null, null);
        clEnqueueFillBuffer(cl.queue, ctrT, Pointer.to(new long[]{0L}),
                Sizeof.cl_ulong, 0, Sizeof.cl_ulong, 0, null, null);

        // 2) Set kernel args
        cl_kernel k = cl.kernel;
        int ai = 0;
        clSetKernelArg(k, ai++, Sizeof.cl_mem, Pointer.to(bufIn));
        clSetKernelArg(k, ai++, Sizeof.cl_uint, Pointer.to(new int[]{inCount}));
        clSetKernelArg(k, ai++, Sizeof.cl_uint, Pointer.to(new int[]{depth}));
        clSetKernelArg(k, ai++, Sizeof.cl_mem, Pointer.to(bufFr));
        clSetKernelArg(k, ai++, Sizeof.cl_mem, Pointer.to(ctrF));
        clSetKernelArg(k, ai++, Sizeof.cl_mem, Pointer.to(bufTX));
        clSetKernelArg(k, ai++, Sizeof.cl_mem, Pointer.to(ctrX));
        clSetKernelArg(k, ai++, Sizeof.cl_mem, Pointer.to(bufTO));
        clSetKernelArg(k, ai++, Sizeof.cl_mem, Pointer.to(ctrO));
        clSetKernelArg(k, ai++, Sizeof.cl_mem, Pointer.to(bufTT));
        clSetKernelArg(k, ai++, Sizeof.cl_mem, Pointer.to(ctrT));

        // 3) Launch
        long localSize  = 128;
        long globalSize = ((totalThds + localSize - 1) / localSize) * localSize;
        clEnqueueNDRangeKernel(cl.queue, k, 1, null,
                new long[]{globalSize}, new long[]{localSize},
                0, null, null);

        // 4) Read back counts
        long[] cntF = new long[1], cntX = new long[1], cntO = new long[1], cntT = new long[1];
        clEnqueueReadBuffer(cl.queue, ctrF, CL_TRUE, 0, Sizeof.cl_ulong, Pointer.to(cntF), 0, null, null);
        clEnqueueReadBuffer(cl.queue, ctrX, CL_TRUE, 0, Sizeof.cl_ulong, Pointer.to(cntX), 0, null, null);
        clEnqueueReadBuffer(cl.queue, ctrO, CL_TRUE, 0, Sizeof.cl_ulong, Pointer.to(cntO), 0, null, null);
        clEnqueueReadBuffer(cl.queue, ctrT, CL_TRUE, 0, Sizeof.cl_ulong, Pointer.to(cntT), 0, null, null);

        // 5) Read back terminal arrays
        long[] termX  = new long[(int)cntX[0]];
        long[] termO  = new long[(int)cntO[0]];
        long[] termTie= new long[(int)cntT[0]];
        if (termX.length  > 0)
            clEnqueueReadBuffer(cl.queue, bufTX, CL_TRUE, 0,
                    termX.length * Sizeof.cl_ulong, Pointer.to(termX), 0, null, null);
        if (termO.length  > 0)
            clEnqueueReadBuffer(cl.queue, bufTO, CL_TRUE, 0,
                    termO.length * Sizeof.cl_ulong, Pointer.to(termO), 0, null, null);
        if (termTie.length> 0)
            clEnqueueReadBuffer(cl.queue, bufTT, CL_TRUE, 0,
                    termTie.length * Sizeof.cl_ulong, Pointer.to(termTie), 0, null, null);

        // 6) Read frontier in <= maxBoards chunks
        List<long[]> frontierChunks = new ArrayList<>();
        long rem = cntF[0], off = 0;
        while (rem > 0) {
            int chunkSize = (int)Math.min(maxBoards, rem);
            long[] chunk = new long[chunkSize];
            clEnqueueReadBuffer(cl.queue, bufFr, CL_TRUE,
                    off * Sizeof.cl_ulong,
                    chunkSize * Sizeof.cl_ulong,
                    Pointer.to(chunk), 0, null, null);
            frontierChunks.add(chunk);
            off += chunkSize;
            rem -= chunkSize;
        }

        // 7) Clean up
        for (cl_mem m : new cl_mem[]{bufIn, bufFr, bufTX, bufTO, bufTT, ctrF, ctrX, ctrO, ctrT}) {
            clReleaseMemObject(m);
        }

        return new Result(frontierChunks, termX, termO, termTie);
    }
}
