package support;

import org.jocl.*;
import java.io.IOException;
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

        // 1) Pick platform & device
        cl_platform_id[] plats = new cl_platform_id[1];
        clGetPlatformIDs(1, plats, null);
        cl_device_id[] devs = new cl_device_id[1];
        clGetDeviceIDs(plats[0], CL_DEVICE_TYPE_GPU, 1, devs, null);
        cl_device_id device = devs[0];

        // 2) Query VRAM & max alloc
        long[] tmp = new long[1];
        clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE,
                Sizeof.cl_ulong, Pointer.to(tmp), null);
        totalMemBytes = tmp[0];
        clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                Sizeof.cl_ulong, Pointer.to(tmp), null);
        maxAllocBytes = tmp[0];
        System.out.printf("VRAM = %,d MB; max alloc = %,d MB%n",
                totalMemBytes/(1024*1024), maxAllocBytes/(1024*1024));

        // 3) Create context + queue
        ctx   = clCreateContext(null, 1, devs, null, null, null);
        queue = clCreateCommandQueue(ctx, device, 0, null);

        // 4) Load .cl from classpath
        String kernelSrc;
        try (var is = getClass().getClassLoader()
                .getResourceAsStream(kernelResourcePath)) {
            if (is == null) throw new IOException("Missing resource: "+kernelResourcePath);
            kernelSrc = new String(is.readAllBytes(), StandardCharsets.UTF_8);
        }

        // 5) Read winLines.txt (space-separated triples) into bitmasks
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

        // 6) Prepend a header that defines WIN_MASKS[N]
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
}
