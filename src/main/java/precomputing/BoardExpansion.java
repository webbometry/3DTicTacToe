package precomputing;

import java.io.*;
import java.math.BigInteger;
import java.nio.charset.StandardCharsets;
import java.util.*;

import org.jocl.*;

import static org.jocl.CL.*;

public class BoardExpansion {
    private static final int BATCH_SIZE = 1_000_000;

    // Paths to CL sources
    private static final String K_HASH = "src/main/resources/HashKernel.cl";
    private static final String K_CANONICAL = "src/main/resources/CanonicalKernel.cl";
    private static final String K_EXPANSION = "src/main/resources/ExpansionKernel.cl";
    private static final String K_WIN = "src/main/resources/WinKernel.cl";
    private static final String K_NEXT_PLAYER = "src/main/resources/NextPlayerKernel.cl";

    // Board constant
    private static final int BOARD_SIZE = 27;

    // OpenCL handles
    private static cl_context context;
    private static cl_command_queue queue;
    private static cl_device_id device;

    // Precomputed data
    private static int[] rotationMaps;   // 24×27
    private static int[] winLines;       // N×3

    // Programs & kernels
    private static cl_program progHash, progCanon, progExpand, progWin, progNext;
    private static cl_kernel kernHash, kernCanon, kernExpand, kernWin, kernNext;

    public static void main(String[] args) throws IOException {
        initCL();
        loadResources();
        buildKernels();

        // Open lookup table for compressed boards
        try (BufferedWriter writer = new BufferedWriter(new FileWriter("gameTree.txt"))) {
            Set<Long> seenHashes = new HashSet<>();
            List<String> current = Collections.singletonList(
                    String.join("", Collections.nCopies(BOARD_SIZE, " "))
            );

            for (int step = 0; step <= BOARD_SIZE; step++) {
                BigInteger theoretical = perm(BigInteger.valueOf(27), step);
                long start = System.currentTimeMillis();

                if (step == 0) {
                    System.out.printf("%nStep %d: Theoretical = %s%n", step, theoretical);
                    System.out.printf("Generated = 1, Difference = %s, Time = 0 ms%n",
                            theoretical.subtract(BigInteger.ONE));
                } else {
                    // 1) canonicalize
                    List<String> canon = runCanonical(current);
                    // 2) hash
                    long[] hashes = runHash(canon);
                    // 3) dedupe
                    List<String> uniqueB = new ArrayList<>();
                    List<Long> uniqueH = new ArrayList<>();
                    for (int i = 0; i < hashes.length; i++) {
                        long h = hashes[i];
                        if (seenHashes.add(h)) {
                            uniqueB.add(canon.get(i));
                            uniqueH.add(h);
                        }
                    }
                    if (uniqueH.isEmpty()) {
                        long elapsed = System.currentTimeMillis() - start;
                        System.out.printf("%nStep %d: Theoretical = %s%n", step, theoretical);
                        System.out.printf("Generated = 0, Difference = %s, Time = %d ms%n",
                                theoretical, elapsed);
                        continue;
                    }
                    // 4) win filter
                    byte[] over = runWin(uniqueH);
                    List<String> survivors = new ArrayList<>();
                    for (int i = 0; i < over.length; i++) {
                        if (over[i] == 0) {
                            survivors.add(uniqueB.get(i));
                        }
                    }
                    // 5) next player
                    byte[] nextArr = runNextPlayer(uniqueH);
                    byte player = nextArr.length > 0 ? nextArr[0] : (byte) 'x';
                    // 6) expand
                    List<String> expanded = runExpansion(survivors, player);

                    // Write compressed expansions to lookup table
                    for (String board : expanded) {
                        long comp = compressBoard(board);
                        writer.write(Long.toUnsignedString(comp));
                        writer.newLine();
                    }
                    writer.flush();

                    long elapsed = System.currentTimeMillis() - start;
                    int generated = uniqueB.size();
                    BigInteger diff = theoretical.subtract(BigInteger.valueOf(generated));
                    System.out.printf("%nStep %d: Theoretical = %s%n", step, theoretical);
                    System.out.printf("Generated = %d, Difference = %s, Time = %d ms%n",
                            generated, diff, elapsed);

                    current = expanded;
                }
            }
        }

        System.out.printf("%nTotal unique canonical states: %d%n", /*replace with actual count*/0);
        releaseCL();
    }

    // ----------------------------
    // OpenCL setup & teardown
    // ----------------------------
    private static void initCL() {
        CL.setExceptionsEnabled(true);
        int[] numPlat = new int[1];
        clGetPlatformIDs(0, null, numPlat);
        cl_platform_id[] plats = new cl_platform_id[numPlat[0]];
        clGetPlatformIDs(plats.length, plats, null);
        int[] numDev = new int[1];
        clGetDeviceIDs(plats[0], CL_DEVICE_TYPE_ALL, 0, null, numDev);
        cl_device_id[] devs = new cl_device_id[numDev[0]];
        clGetDeviceIDs(plats[0], CL_DEVICE_TYPE_ALL, devs.length, devs, null);
        device = devs[0];
        context = clCreateContext(null, 1, new cl_device_id[]{device}, null, null, null);
        cl_queue_properties props = new cl_queue_properties();
        queue = clCreateCommandQueueWithProperties(context, device, props, null);
    }

    private static void releaseCL() {
        clReleaseKernel(kernHash);
        clReleaseKernel(kernCanon);
        clReleaseKernel(kernExpand);
        clReleaseKernel(kernWin);
        clReleaseKernel(kernNext);
        clReleaseProgram(progHash);
        clReleaseProgram(progCanon);
        clReleaseProgram(progExpand);
        clReleaseProgram(progWin);
        clReleaseProgram(progNext);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
    }

    // ----------------------------
    // Load rotationMaps.txt & winLines.txt
    // ----------------------------
    private static void loadResources() throws IOException {
        rotationMaps = loadIntMatrix("rotationMaps.txt", 24, BOARD_SIZE);
        winLines = loadIntMatrix("winLines.txt", -1, 3);
        if (winLines.length == 0) {
            throw new IllegalStateException(
                    "winLines.txt loaded 0 entries—check that it’s in the working directory and each line has 3 ints."
            );
        }
    }

    private static int[] loadIntMatrix(String path, int rows, int cols) throws IOException {
        List<Integer> all = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line;
            int expected = cols;
            while ((line = br.readLine()) != null) {
                String[] tok = line.trim().split("\\s+");
                if (cols < 0) expected = tok.length;
                if (tok.length != expected) continue;
                for (String s : tok) all.add(Integer.parseInt(s));
            }
        }
        int[] out = new int[all.size()];
        for (int i = 0; i < out.length; i++) out[i] = all.get(i);
        return out;
    }

    // ----------------------------
    // Build programs & kernels
    // ----------------------------
    private static void buildKernels() throws IOException {
        progHash = buildProg(K_HASH);
        progCanon = buildProg(K_CANONICAL);
        progExpand = buildProg(K_EXPANSION);
        progWin = buildProg(K_WIN);
        progNext = buildProg(K_NEXT_PLAYER);

        kernHash = clCreateKernel(progHash, "hashBoard", null);
        kernCanon = clCreateKernel(progCanon, "canonicalizeBoard", null);
        kernExpand = clCreateKernel(progExpand, "expandBoards", null);
        kernWin = clCreateKernel(progWin, "checkWin", null);
        kernNext = clCreateKernel(progNext, "nextPlayer", null);
    }

    private static cl_program buildProg(String path) throws IOException {
        String src = new String(java.nio.file.Files.readAllBytes(
                new File(path).toPath()), StandardCharsets.UTF_8);
        cl_program p = clCreateProgramWithSource(context, 1, new String[]{src}, null, null);
        clBuildProgram(p, 0, null, null, null, null);
        return p;
    }


    // ----------------------------
    // Kernel runners
    // ----------------------------
    private static List<String> runCanonical(List<String> boards) {
        int n = boards.size();
        byte[] in = flatten(boards);
        byte[] out = new byte[n * BOARD_SIZE];

        cl_mem mbIn = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_char * in.length, Pointer.to(in), null);
        cl_mem mbOut = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                Sizeof.cl_char * out.length, null, null);
        cl_mem mbRot = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_int * rotationMaps.length, Pointer.to(rotationMaps), null);

        clSetKernelArg(kernCanon, 0, Sizeof.cl_mem, Pointer.to(mbIn));
        clSetKernelArg(kernCanon, 1, Sizeof.cl_mem, Pointer.to(mbOut));
        clSetKernelArg(kernCanon, 2, Sizeof.cl_mem, Pointer.to(mbRot));
        clSetKernelArg(kernCanon, 3, Sizeof.cl_int, Pointer.to(new int[]{n}));

        clEnqueueNDRangeKernel(queue, kernCanon, 1, null, new long[]{n}, null, 0, null, null);
        clEnqueueReadBuffer(queue, mbOut, CL_TRUE, 0, out.length, Pointer.to(out), 0, null, null);

        clReleaseMemObject(mbIn);
        clReleaseMemObject(mbOut);
        clReleaseMemObject(mbRot);

        return unflatten(out);
    }

    private static long[] runHash(List<String> boards) {
        int n = boards.size();
        byte[] in = flatten(boards);
        long[] out = new long[n];

        // Input buffer with COPY_HOST_PTR
        cl_mem mbIn = clCreateBuffer(
                context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_char * in.length,
                Pointer.to(in),
                null
        );
        // Output buffer: no host ptr on a WRITE_ONLY buffer
        cl_mem mbOut = clCreateBuffer(
                context,
                CL_MEM_WRITE_ONLY,
                Sizeof.cl_ulong * n,
                null,
                null
        );

        clSetKernelArg(kernHash, 0, Sizeof.cl_mem, Pointer.to(mbIn));
        clSetKernelArg(kernHash, 1, Sizeof.cl_mem, Pointer.to(mbOut));
        clSetKernelArg(kernHash, 2, Sizeof.cl_int, Pointer.to(new int[]{n}));

        clEnqueueNDRangeKernel(queue, kernHash, 1, null, new long[]{n}, null, 0, null, null);
        clEnqueueReadBuffer(queue, mbOut, CL_TRUE, 0, Sizeof.cl_ulong * n, Pointer.to(out), 0, null, null);

        clReleaseMemObject(mbIn);
        clReleaseMemObject(mbOut);

        return out;
    }

    private static byte[] runWin(List<Long> hashes) {
        int n = hashes.size();
        if (n == 0) {
            return new byte[0];
        }
        // Prepare host arrays
        long[] in = new long[n];
        for (int i = 0; i < n; i++) in[i] = hashes.get(i);
        byte[] over = new byte[n];
        byte[] winnerB = new byte[n];

        // Create buffers
        cl_mem mbIn = clCreateBuffer(
                context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_ulong * n,
                Pointer.to(in),
                null
        );
        cl_mem mbLines = clCreateBuffer(
                context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_int * winLines.length,
                Pointer.to(winLines),
                null
        );
        cl_mem mbOver = clCreateBuffer(
                context,
                CL_MEM_WRITE_ONLY,
                Sizeof.cl_uchar * n,
                null,
                null
        );
        cl_mem mbWin = clCreateBuffer(
                context,
                CL_MEM_WRITE_ONLY,
                Sizeof.cl_uchar * n,
                null,
                null
        );

        // Set args (now all five)
        clSetKernelArg(kernWin, 0, Sizeof.cl_mem, Pointer.to(mbIn));
        clSetKernelArg(kernWin, 1, Sizeof.cl_mem, Pointer.to(mbLines));
        clSetKernelArg(kernWin, 2, Sizeof.cl_int, Pointer.to(new int[]{winLines.length / 3}));
        clSetKernelArg(kernWin, 3, Sizeof.cl_mem, Pointer.to(mbOver));
        clSetKernelArg(kernWin, 4, Sizeof.cl_mem, Pointer.to(mbWin));

        // Enqueue & read back only gameOver
        clEnqueueNDRangeKernel(queue, kernWin, 1, null, new long[]{n}, null, 0, null, null);
        clEnqueueReadBuffer(queue, mbOver, CL_TRUE, 0, over.length, Pointer.to(over), 0, null, null);

        // Clean up
        clReleaseMemObject(mbIn);
        clReleaseMemObject(mbLines);
        clReleaseMemObject(mbOver);
        clReleaseMemObject(mbWin);

        return over;
    }


    private static byte[] runNextPlayer(List<Long> hashes) {
        int n = hashes.size();
        if (n == 0) {
            return new byte[0];
        }

        long[] in = new long[n];
        for (int i = 0; i < n; i++) in[i] = hashes.get(i);
        byte[] out = new byte[n];

        cl_mem mbIn = clCreateBuffer(
                context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_ulong * n,
                Pointer.to(in),
                null
        );
        // WRITE_ONLY ⇒ host ptr must be null
        cl_mem mbOut = clCreateBuffer(
                context,
                CL_MEM_WRITE_ONLY,
                Sizeof.cl_uchar * n,
                null,
                null
        );

        clSetKernelArg(kernNext, 0, Sizeof.cl_mem, Pointer.to(mbIn));
        clSetKernelArg(kernNext, 1, Sizeof.cl_int, Pointer.to(new int[]{n}));
        clSetKernelArg(kernNext, 2, Sizeof.cl_mem, Pointer.to(mbOut));

        clEnqueueNDRangeKernel(queue, kernNext, 1, null, new long[]{n}, null, 0, null, null);
        clEnqueueReadBuffer(queue, mbOut, CL_TRUE, 0, Sizeof.cl_uchar * n, Pointer.to(out), 0, null, null);

        clReleaseMemObject(mbIn);
        clReleaseMemObject(mbOut);

        return out;
    }

    private static List<String> runExpansion(List<String> boards, byte nextPlayer) {
        int n = boards.size();
        if (n == 0) {
            return Collections.emptyList();
        }

        // Prevent int overflow:
        long maxBoardsLong = (long) n * BOARD_SIZE;
        long totalBytesLong = maxBoardsLong * BOARD_SIZE;
        if (totalBytesLong > Integer.MAX_VALUE) {
            throw new IllegalStateException(
                    "Too many boards to expand: " + n +
                            " ⇒ needs " + totalBytesLong + " bytes"
            );
        }

        // Flatten boards into a single byte array
        byte[] inFlat = flatten(boards);

        // Worst‐case output buffer size
        int maxBoards = n * BOARD_SIZE;
        byte[] outFlat = new byte[maxBoards * BOARD_SIZE];

        // Atomic counter init
        int[] zero = new int[]{0};

        // Create buffers
        cl_mem mbIn = clCreateBuffer(context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_char * inFlat.length,
                Pointer.to(inFlat), null);

        cl_mem mbOut = clCreateBuffer(context,
                CL_MEM_WRITE_ONLY,
                Sizeof.cl_char * outFlat.length,
                null, null);

        cl_mem mbCount = clCreateBuffer(context,
                CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_int,
                Pointer.to(zero), null);

        // Set arguments:
        //   0: input boards
        clSetKernelArg(kernExpand, 0, Sizeof.cl_mem, Pointer.to(mbIn));
        //   1: output boards
        clSetKernelArg(kernExpand, 1, Sizeof.cl_mem, Pointer.to(mbOut));
        //   2: atomic counter
        clSetKernelArg(kernExpand, 2, Sizeof.cl_mem, Pointer.to(mbCount));
        //   3: single nextPlayer char
        clSetKernelArg(kernExpand, 3, Sizeof.cl_uchar, Pointer.to(new byte[]{nextPlayer}));
        //   4: board count
        clSetKernelArg(kernExpand, 4, Sizeof.cl_int, Pointer.to(new int[]{n}));

        // Launch
        clEnqueueNDRangeKernel(queue, kernExpand, 1, null,
                new long[]{n}, null, 0, null, null);

        // Read back how many boards were generated
        int[] cnt = new int[1];
        clEnqueueReadBuffer(queue, mbCount, CL_TRUE, 0,
                Sizeof.cl_int, Pointer.to(cnt), 0, null, null);
        int outCount = cnt[0];

        // Read back the actual boards
        clEnqueueReadBuffer(queue, mbOut, CL_TRUE, 0,
                Sizeof.cl_char * outCount * BOARD_SIZE,
                Pointer.to(outFlat), 0, null, null);

        // Cleanup
        clReleaseMemObject(mbIn);
        clReleaseMemObject(mbOut);
        clReleaseMemObject(mbCount);

        // Build the List<String> to return
        List<String> result = new ArrayList<>(outCount);
        for (int i = 0; i < outCount; i++) {
            result.add(new String(
                    outFlat, i * BOARD_SIZE, BOARD_SIZE, StandardCharsets.US_ASCII
            ));
        }
        return result;
    }

    // ----------------------------
    // File Writing
    // ----------------------------
    private static void runExpansionStreamToDisk(List<String> boards,
                                                 byte nextPlayer,
                                                 int step) throws IOException {
        if (boards.isEmpty()) return;

        // File for this step's expansions
        File outFile = new File("expansions_step_" + step + ".bin");
        try (DataOutputStream dos = new DataOutputStream(
                new BufferedOutputStream(new FileOutputStream(outFile)))) {
            int total = boards.size();
            for (int offset = 0; offset < total; offset += BATCH_SIZE) {
                int end = Math.min(total, offset + BATCH_SIZE);
                List<String> batch = boards.subList(offset, end);

                // === Flatten & prepare CL buffers exactly as before, but only for 'batch' ===
                byte[] inFlat = flatten(batch);
                int n = batch.size();
                int maxBoards = n * BOARD_SIZE;
                byte[] outFlat = new byte[maxBoards * BOARD_SIZE];
                int[] zero = new int[]{0};

                cl_mem mbIn = clCreateBuffer(context,
                        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        Sizeof.cl_char * inFlat.length,
                        Pointer.to(inFlat), null);
                cl_mem mbOut = clCreateBuffer(context,
                        CL_MEM_WRITE_ONLY,
                        Sizeof.cl_char * outFlat.length,
                        null, null);
                cl_mem mbCt = clCreateBuffer(context,
                        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                        Sizeof.cl_int,
                        Pointer.to(zero), null);

                clSetKernelArg(kernExpand, 0, Sizeof.cl_mem, Pointer.to(mbIn));
                clSetKernelArg(kernExpand, 1, Sizeof.cl_mem, Pointer.to(mbOut));
                clSetKernelArg(kernExpand, 2, Sizeof.cl_mem, Pointer.to(mbCt));
                clSetKernelArg(kernExpand, 3, Sizeof.cl_uchar, Pointer.to(new byte[]{nextPlayer}));
                clSetKernelArg(kernExpand, 4, Sizeof.cl_int, Pointer.to(new int[]{n}));

                // launch & read back counters
                clEnqueueNDRangeKernel(queue, kernExpand, 1, null,
                        new long[]{n}, null, 0, null, null);
                int[] cnt = new int[1];
                clEnqueueReadBuffer(queue, mbCt, CL_TRUE, 0,
                        Sizeof.cl_int, Pointer.to(cnt), 0, null, null);
                int outCount = cnt[0];

                // read out the boards
                clEnqueueReadBuffer(queue, mbOut, CL_TRUE, 0,
                        Sizeof.cl_char * outCount * BOARD_SIZE,
                        Pointer.to(outFlat), 0, null, null);

                // clean up CL
                clReleaseMemObject(mbIn);
                clReleaseMemObject(mbOut);
                clReleaseMemObject(mbCt);

                // compress & write each board to disk as a long
                for (int i = 0; i < outCount; i++) {
                    String b = new String(
                            outFlat, i * BOARD_SIZE, BOARD_SIZE, StandardCharsets.US_ASCII);
                    long compressed = compressBoard(b);
                    dos.writeLong(compressed);
                }
            }
        }
    }

    // ----------------------------
    // Utilities
    // ----------------------------
    private static byte[] flatten(List<String> boards) {
        int n = boards.size();
        byte[] arr = new byte[n * BOARD_SIZE];
        for (int i = 0; i < n; i++) {
            byte[] b = boards.get(i).getBytes(StandardCharsets.US_ASCII);
            System.arraycopy(b, 0, arr, i * BOARD_SIZE, BOARD_SIZE);
        }
        return arr;
    }

    private static List<String> unflatten(byte[] flat) {
        int n = flat.length / BOARD_SIZE;
        List<String> list = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            list.add(new String(flat, i * BOARD_SIZE, BOARD_SIZE, StandardCharsets.US_ASCII));
        }
        return list;
    }

    private static BigInteger perm(BigInteger n, int k) {
        BigInteger r = BigInteger.ONE;
        for (int i = 0; i < k; i++) r = r.multiply(n.subtract(BigInteger.valueOf(i)));
        return r;
    }

    private static long compressBoard(String board) {
        long bits = 0L;
        for (int i = 0; i < BOARD_SIZE; i++) {
            char c = board.charAt(i);
            if (c == 'x') bits |= (1L << i);
            else if (c == 'o') bits |= (1L << (i + BOARD_SIZE));
        }
        return bits;
    }

    private static String decompressBoard(long bits) {
        char[] buf = new char[BOARD_SIZE];
        for (int i = 0; i < BOARD_SIZE; i++) {
            boolean hasX = ((bits >>> i) & 1L) != 0;
            boolean hasO = ((bits >>> (i + BOARD_SIZE)) & 1L) != 0;
            buf[i] = hasX ? 'x' : hasO ? 'o' : ' ';
        }
        return new String(buf);
    }

    private static List<String> loadCompressedFromDisk(int step) throws IOException {
        File inFile = new File("expansions_step_" + step + ".bin");
        List<String> boards = new ArrayList<>();
        try (DataInputStream dis = new DataInputStream(
                new BufferedInputStream(new FileInputStream(inFile)))) {
            while (dis.available() > 0) {
                long bits = dis.readLong();
                boards.add(decompressBoard(bits));
            }
        }
        return boards;
    }
}
