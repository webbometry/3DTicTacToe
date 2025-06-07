package support;

import org.jocl.*;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

import static org.jocl.CL.*;

public class CLContext {
    public final cl_context       ctx;
    public final cl_command_queue queue;
    public final cl_program       program;
    public final cl_kernel        kernel;

    public final long totalMemBytes;
    public final long maxAllocBytes;

    /**
     * @param kernelResourcePath path on classpath (e.g. "cl/expand_and_classify.cl")
     */
    public CLContext(String kernelResourcePath) throws IOException {
        CL.setExceptionsEnabled(true);

        // 1) Pick the NVIDIA GPU if present
        cl_platform_id[] platforms = new cl_platform_id[1];
        clGetPlatformIDs(1, platforms, null);

        cl_device_id selected = null;
        // Query how many GPU devices
        int[] gpuCount = new int[1];
        clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, null, gpuCount);
        cl_device_id[] gpus = new cl_device_id[gpuCount[0]];
        clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, gpus.length, gpus, null);

        for (cl_device_id d : gpus) {
            byte[] buf = new byte[1024];
            clGetDeviceInfo(d, CL_DEVICE_VENDOR, buf.length, Pointer.to(buf), null);
            String vendor = new String(buf, StandardCharsets.UTF_8).trim();
            if (vendor.toLowerCase().contains("nvidia")) {
                selected = d;
                break;
            }
        }
        if (selected == null && gpus.length > 0) {
            selected = gpus[0];
            System.out.println("Warning: NVIDIA GPU not found, using first GPU: " +
                    deviceName(gpus[0]));
        }
        if (selected == null) {
            throw new IllegalStateException("No OpenCL GPU devices found");
        }

        // 2) Query VRAM & max alloc
        long[] tmp = new long[1];
        clGetDeviceInfo(selected, CL_DEVICE_GLOBAL_MEM_SIZE, Sizeof.cl_ulong, Pointer.to(tmp), null);
        totalMemBytes = tmp[0];
        clGetDeviceInfo(selected, CL_DEVICE_MAX_MEM_ALLOC_SIZE, Sizeof.cl_ulong, Pointer.to(tmp), null);
        maxAllocBytes = tmp[0];
        System.out.printf("Using device %s%n", deviceName(selected));
        System.out.printf("  VRAM = %,d MB; max single alloc = %,d MB%n",
                totalMemBytes/(1024*1024), maxAllocBytes/(1024*1024));

        // 3) Create context & queue
        ctx   = clCreateContext(null, 1, new cl_device_id[]{selected}, null, null, null);
        queue = clCreateCommandQueue(ctx, selected, 0, null);

        // 4) Load kernel source from classpath
        String kernelSrc;
        try (InputStream is = getClass()
                .getClassLoader()
                .getResourceAsStream(kernelResourcePath)) {
            if (is == null) {
                throw new IOException("Kernel not found on classpath: " + kernelResourcePath);
            }
            byte[] bytes = is.readAllBytes();
            kernelSrc = new String(bytes, StandardCharsets.UTF_8);
        }

        // 5) Read winLines.txt (space-separated indices) into masks[]
        List<String> lines = Files.readAllLines(
                Paths.get("C:\\Users\\webbometric\\Documents\\GitHub\\3DTicTacToe\\src\\main\\data\\winLines.txt"),
                StandardCharsets.UTF_8
        );
        int    N   = lines.size();
        long[] masks = new long[N];
        for (int i = 0; i < N; i++) {
            String[] parts = lines.get(i).trim().split("\\s+");
            long m = 0L;
            for (String p : parts) {
                int idx = Integer.parseInt(p);
                m |= (1L << idx);
            }
            masks[i] = m;
        }

        // 6) Build header with WIN_MASKS[N]
        StringBuilder header = new StringBuilder();
        header.append("__constant ulong WIN_MASKS[").append(N).append("] = {");
        for (int i = 0; i < N; i++) {
            header.append("0x")
                    .append(Long.toHexString(masks[i]))
                    .append("UL")
                    .append(i+1 < N ? "," : "");
        }
        header.append("};\n\n");

        // 7) Compile with injected header
        String fullSrc = header + kernelSrc;
        program = clCreateProgramWithSource(ctx, 1, new String[]{ fullSrc }, null, null);
        clBuildProgram(program, 0, null, null, null, null);

        // 8) Create kernel
        kernel = clCreateKernel(program, "expand_and_classify", null);
    }

    /** Helper to query device name */
    private static String deviceName(cl_device_id d) {
        byte[] buf = new byte[1024];
        clGetDeviceInfo(d, CL_DEVICE_NAME, buf.length, Pointer.to(buf), null);
        return new String(buf, StandardCharsets.UTF_8).trim();
    }
}
