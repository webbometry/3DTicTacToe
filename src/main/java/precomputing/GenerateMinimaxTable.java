package precomputing;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

import org.jocl.*;

import static org.jocl.CL.*;

public class GenerateMinimaxTable {
    private static final int MAX_POS = 27;
    private static final int BATCH_SIZE = 100_000;

    // JOCL handles
    private static cl_context context;
    private static cl_command_queue queue;
    private static cl_device_id device;

    // Programs & kernels
    private static cl_program progExpand, progCanon, progWin;
    private static cl_kernel kernExpand, kernCanon, kernWin;

    // Precomputed data
    private static int[] rotationMaps;
    private static byte[] winUpperChars, winLowerChars;
    private static cl_mem mbWinUpper, mbWinLower;

    public static void main(String[] args) throws IOException {
        initCL();

        // Load rotation maps (24×27 ints)
        rotationMaps = loadIntMatrix("/precomputed/rotationMaps.txt", 24, MAX_POS);

        // Load win‑lines (W×3 chars)
        winUpperChars = loadCharMatrix("/precomputed/winLinesUpper.txt", 3);
        winLowerChars = loadCharMatrix("/precomputed/winLinesLower.txt", 3);
        int W = winUpperChars.length / 3;
        if (winLowerChars.length != W * 3)
            throw new IllegalStateException("winLinesUpper/lower length mismatch");

        // Build kernels
        progExpand = buildProgram("/ExpansionKernel.cl");
        kernExpand = clCreateKernel(progExpand, "expandBoards", null);
        progCanon = buildProgram("/CanonicalizationKernel.cl");
        kernCanon = clCreateKernel(progCanon, "canonicalize", null);
        progWin = buildProgram("/CheckWinKernel.cl");
        kernWin = clCreateKernel(progWin, "checkWin", null);

        // Create CL buffers for win‑lines
        mbWinUpper = clCreateBuffer(context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_char * winUpperChars.length,
                Pointer.to(winUpperChars), null);
        mbWinLower = clCreateBuffer(context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_char * winLowerChars.length,
                Pointer.to(winLowerChars), null);

        // See if we already have step 8 frontier cached
        Path cache8 = Paths.get("frontier_step8.txt");
        List<String> frontier;
        int startStep;
        if (Files.exists(cache8)) {
            // load it and skip to step 9
            frontier = Files.readAllLines(cache8, StandardCharsets.US_ASCII);
            startStep = 9;
            System.out.printf("Loaded %d boards from frontier_step8.txt, skipping to step 9%n",
                    frontier.size());
        } else {
            frontier = Collections.singletonList("");
            startStep = 1;
        }

        // Generation loop
        for (int step = startStep; step <= MAX_POS; step++) {
            long t0 = System.currentTimeMillis();

            // 1) Expand + canonicalize → temp file
            Path expPath = Paths.get("exp_step_" + step + ".txt");
            Files.deleteIfExists(expPath);
            try (BufferedWriter expOut = Files.newBufferedWriter(
                    expPath, StandardCharsets.US_ASCII,
                    StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)) {
                for (int off = 0; off < frontier.size(); off += BATCH_SIZE) {
                    int end = Math.min(frontier.size(), off + BATCH_SIZE);
                    List<String> batch = frontier.subList(off, end);

                    // expand
                    List<String> expanded = runExpansion(batch, step);
                    // canonicalize
                    List<String> canoned = runCanonical(expanded, step);

                    // stream out
                    for (String b : canoned) {
                        expOut.write(b);
                        expOut.newLine();
                    }
                }
            }

            // 2) Win‑check → next frontier & terminals
            List<String> nextFrontier = new ArrayList<>();
            Path termPath = Paths.get("minimax_terminal.txt");
            Files.createDirectories(termPath.getParent() == null
                    ? Paths.get(".") : termPath.getParent());
            try (BufferedReader expIn = Files.newBufferedReader(expPath, StandardCharsets.US_ASCII);
                 BufferedWriter termOut = Files.newBufferedWriter(
                         termPath, StandardCharsets.US_ASCII,
                         StandardOpenOption.CREATE, StandardOpenOption.APPEND)) {
                while (true) {
                    List<String> chunk = new ArrayList<>(BATCH_SIZE);
                    String line;
                    for (int i = 0; i < BATCH_SIZE && (line = expIn.readLine()) != null; i++) {
                        chunk.add(line);
                    }
                    if (chunk.isEmpty()) break;

                    int[] stats = runCheckWin(chunk, step, W);
                    for (int i = 0; i < chunk.size(); i++) {
                        String b = chunk.get(i);
                        if (stats[i] == 0) {
                            nextFrontier.add(b);
                        } else if (stats[i] == 1) {
                            termOut.write(b + "+");
                            termOut.newLine();
                        } else {
                            termOut.write(b);
                            termOut.newLine();
                        }
                    }
                }
            }
            Files.deleteIfExists(expPath);

            // 3) Overwrite minimax.txt with survivors
            Path mmPath = Paths.get("minimax.txt");
            try (BufferedWriter mmOut = Files.newBufferedWriter(
                    mmPath, StandardCharsets.US_ASCII,
                    StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)) {
                for (String b : nextFrontier) {
                    mmOut.write(b);
                    mmOut.newLine();
                }
            }

            // 4) If this was step 8, cache the frontier
            if (step == 8 && !Files.exists(cache8)) {
                Files.write(cache8,
                        nextFrontier,
                        StandardCharsets.US_ASCII,
                        StandardOpenOption.CREATE_NEW);
                System.out.printf("Wrote %d boards to frontier_step8.txt%n", nextFrontier.size());
            }

            frontier = nextFrontier;
            long dt = System.currentTimeMillis() - t0;
            System.out.printf("Step %d → surv=%d, time=%d ms%n",
                    step, frontier.size(), dt);
        }

        releaseCL();
    }

    // ----------------------------
    // OpenCL & resource helpers
    // ----------------------------
    private static void initCL() {
        CL.setExceptionsEnabled(true);
        int[] np = new int[1];
        clGetPlatformIDs(0, null, np);
        cl_platform_id[] plats = new cl_platform_id[np[0]];
        clGetPlatformIDs(plats.length, plats, null);
        int[] nd = new int[1];
        clGetDeviceIDs(plats[0], CL_DEVICE_TYPE_ALL, 0, null, nd);
        cl_device_id[] devs = new cl_device_id[nd[0]];
        clGetDeviceIDs(plats[0], CL_DEVICE_TYPE_ALL, devs.length, devs, null);
        device = devs[0];
        context = clCreateContext(null, 1, new cl_device_id[]{device}, null, null, null);
        queue = clCreateCommandQueueWithProperties(context, device, new cl_queue_properties(), null);
    }

    private static void releaseCL() {
        clReleaseKernel(kernExpand);
        clReleaseKernel(kernCanon);
        clReleaseKernel(kernWin);
        clReleaseProgram(progExpand);
        clReleaseProgram(progCanon);
        clReleaseProgram(progWin);
        clReleaseMemObject(mbWinUpper);
        clReleaseMemObject(mbWinLower);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
    }

    private static int[] loadIntMatrix(String resource, int rows, int cols) throws IOException {
        List<Integer> L = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(
                new InputStreamReader(
                        GenerateMinimaxTable.class.getResourceAsStream(resource),
                        StandardCharsets.US_ASCII))) {
            String line;
            int expected = cols;
            while ((line = br.readLine()) != null) {
                String[] tok = line.trim().split("\\s+");
                if (cols < 0) expected = tok.length;
                if (tok.length != expected) continue;
                for (String t : tok) L.add(Integer.parseInt(t));
            }
        }
        int[] out = new int[L.size()];
        for (int i = 0; i < out.length; i++) out[i] = L.get(i);
        return out;
    }

    private static byte[] loadCharMatrix(String resource, int cols) throws IOException {
        List<Byte> L = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(
                new InputStreamReader(
                        GenerateMinimaxTable.class.getResourceAsStream(resource),
                        StandardCharsets.US_ASCII))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] tok = line.trim().split("\\s+");
                if (tok.length != cols) continue;
                for (String t : tok) {
                    if (t.length() != 1) throw new IOException("Bad token:" + t);
                    L.add((byte) t.charAt(0));
                }
            }
        }
        byte[] out = new byte[L.size()];
        for (int i = 0; i < out.length; i++) out[i] = L.get(i);
        return out;
    }

    private static cl_program buildProgram(String resource) throws IOException {
        String src;
        try (InputStream in = GenerateMinimaxTable.class.getResourceAsStream(resource)) {
            src = new String(in.readAllBytes(), StandardCharsets.US_ASCII);
        }
        cl_program p = clCreateProgramWithSource(context, 1, new String[]{src}, null, null);
        clBuildProgram(p, 0, null, null, null, null);
        return p;
    }

    // ----------------------------
    // GPU kernel runners
    // ----------------------------
    private static List<String> runExpansion(List<String> boards, int step) {
        // Step 1: generate the 27 first‑moves (X’s turn) in Java, no OpenCL call
        if (step == 1) {
            List<String> first = new ArrayList<>(27);
            // 'A' through 'Z'
            for (char c = 'A'; c <= 'Z'; c++) {
                first.add(String.valueOf(c));
            }
            // the 27th spot
            first.add(".");
            return first;
        }

        // All later steps (step >= 2) use the kernel as before
        int n = boards.size();
        if (n == 0) {
            return Collections.emptyList();
        }

        int prevLen = step - 1;
        // flatten previous sequences (length=prevLen)
        byte[] inFlat = new byte[n * prevLen];
        for (int i = 0; i < n; i++) {
            byte[] b = boards.get(i).getBytes(StandardCharsets.US_ASCII);
            System.arraycopy(b, 0, inFlat, i * prevLen, prevLen);
        }

        // worst‐case output size
        int maxOut = n * MAX_POS * step;
        byte[] outFlat = new byte[maxOut];
        int[] zero = new int[]{0};

        cl_mem mbIn = clCreateBuffer(context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_char * inFlat.length,
                Pointer.to(inFlat), null);
        cl_mem mbOut = clCreateBuffer(context,
                CL_MEM_WRITE_ONLY,
                Sizeof.cl_char * outFlat.length, null, null);
        cl_mem mbCt = clCreateBuffer(context,
                CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_int, Pointer.to(zero), null);

        byte nextPlayer = ((step - 1) & 1) == 0 ? (byte) 0 : (byte) 1;

        clSetKernelArg(kernExpand, 0, Sizeof.cl_mem, Pointer.to(mbIn));
        clSetKernelArg(kernExpand, 1, Sizeof.cl_mem, Pointer.to(mbOut));
        clSetKernelArg(kernExpand, 2, Sizeof.cl_mem, Pointer.to(mbCt));
        clSetKernelArg(kernExpand, 3, Sizeof.cl_int, Pointer.to(new int[]{prevLen}));
        clSetKernelArg(kernExpand, 4, Sizeof.cl_uchar, Pointer.to(new byte[]{nextPlayer}));
        clSetKernelArg(kernExpand, 5, Sizeof.cl_int, Pointer.to(new int[]{n}));

        clEnqueueNDRangeKernel(queue, kernExpand, 1, null, new long[]{n}, null, 0, null, null);

        int[] cnt = new int[1];
        clEnqueueReadBuffer(queue, mbCt, CL_TRUE, 0, Sizeof.cl_int, Pointer.to(cnt), 0, null, null);
        int total = cnt[0];

        clEnqueueReadBuffer(queue, mbOut, CL_TRUE, 0,
                Sizeof.cl_char * total * step,
                Pointer.to(outFlat), 0, null, null);

        clReleaseMemObject(mbIn);
        clReleaseMemObject(mbOut);
        clReleaseMemObject(mbCt);

        List<String> result = new ArrayList<>(total);
        for (int i = 0; i < total; i++) {
            result.add(new String(outFlat, i * step, step, StandardCharsets.US_ASCII));
        }
        return result;
    }


    private static List<String> runCanonical(List<String> boards, int step) {
        int n = boards.size();
        if (n == 0) return Collections.emptyList();

        byte[] inFlat = new byte[n * step];
        for (int i = 0; i < n; i++) {
            System.arraycopy(boards.get(i).getBytes(StandardCharsets.US_ASCII),
                    0, inFlat, i * step, step);
        }
        byte[] flags = new byte[n];

        cl_mem mbIn = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_char * inFlat.length, Pointer.to(inFlat), null);
        cl_mem mbRot = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_int * rotationMaps.length, Pointer.to(rotationMaps), null);
        cl_mem mbOut = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                Sizeof.cl_uchar * n, null, null);

        clSetKernelArg(kernCanon, 0, Sizeof.cl_mem, Pointer.to(mbIn));
        clSetKernelArg(kernCanon, 1, Sizeof.cl_mem, Pointer.to(mbRot));
        clSetKernelArg(kernCanon, 2, Sizeof.cl_int, Pointer.to(new int[]{n}));
        clSetKernelArg(kernCanon, 3, Sizeof.cl_int, Pointer.to(new int[]{step}));
        clSetKernelArg(kernCanon, 4, Sizeof.cl_mem, Pointer.to(mbOut));

        clEnqueueNDRangeKernel(queue, kernCanon, 1, null, new long[]{n}, null, 0, null, null);
        clEnqueueReadBuffer(queue, mbOut, CL_TRUE, 0, Sizeof.cl_uchar * n, Pointer.to(flags), 0, null, null);

        clReleaseMemObject(mbIn);
        clReleaseMemObject(mbRot);
        clReleaseMemObject(mbOut);

        List<String> canon = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            if (flags[i] != 0) canon.add(boards.get(i));
        }
        return canon;
    }

    private static int[] runCheckWin(List<String> boards, int step, int W) {
        int n = boards.size();
        if (n == 0) return new int[0];

        byte[] inFlat = new byte[n * step];
        for (int i = 0; i < n; i++) {
            System.arraycopy(boards.get(i).getBytes(StandardCharsets.US_ASCII),
                    0, inFlat, i * step, step);
        }
        int[] outStat = new int[n];

        cl_mem mbIn = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_char * inFlat.length, Pointer.to(inFlat), null);
        cl_mem mbStatus = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                Sizeof.cl_int * n, null, null);

        // set args: inSeqs, winUpper, winLower, W, N, step, statusOut
        clSetKernelArg(kernWin, 0, Sizeof.cl_mem, Pointer.to(mbIn));
        clSetKernelArg(kernWin, 1, Sizeof.cl_mem, Pointer.to(mbWinUpper));
        clSetKernelArg(kernWin, 2, Sizeof.cl_mem, Pointer.to(mbWinLower));
        clSetKernelArg(kernWin, 3, Sizeof.cl_int, Pointer.to(new int[]{W}));
        clSetKernelArg(kernWin, 4, Sizeof.cl_int, Pointer.to(new int[]{n}));
        clSetKernelArg(kernWin, 5, Sizeof.cl_int, Pointer.to(new int[]{step}));
        clSetKernelArg(kernWin, 6, Sizeof.cl_mem, Pointer.to(mbStatus));

        clEnqueueNDRangeKernel(queue, kernWin, 1, null, new long[]{n}, null, 0, null, null);
        clEnqueueReadBuffer(queue, mbStatus, CL_TRUE, 0, Sizeof.cl_int * n, Pointer.to(outStat), 0, null, null);

        clReleaseMemObject(mbIn);
        clReleaseMemObject(mbStatus);

        return outStat;
    }
}
