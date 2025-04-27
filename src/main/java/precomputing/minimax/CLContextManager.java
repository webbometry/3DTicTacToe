package precomputing.minimax;

import static org.jocl.CL.*;

import org.jocl.*;

public class CLContextManager {
    private static cl_context context;
    private static cl_command_queue queue;

    static {
        CL.setExceptionsEnabled(true);
        // pick first platform & first GPU device
        cl_platform_id[] platforms = new cl_platform_id[1];
        clGetPlatformIDs(platforms.length, platforms, null);
        cl_device_id[] devices = new cl_device_id[1];
        clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, devices.length, devices, null);

        cl_context_properties props = new cl_context_properties();
        props.addProperty(CL_CONTEXT_PLATFORM, platforms[0]);
        context = clCreateContext(props, 1, devices, null, null, null);
        queue = clCreateCommandQueueWithProperties(context, devices[0], null, null);
    }

    /**
     * @return shared OpenCL context
     */
    public static cl_context getContext() {
        return context;
    }

    /**
     * @return shared OpenCL command queue
     */
    public static cl_command_queue getQueue() {
        return queue;
    }

    /**
     * Build an OpenCL program by filename (must live in src/main/resources/CL/).
     * You can pass "TurnDetection.cl" or even full paths; we strip to filename.
     */
    public static cl_program buildProgram(String resourceName) {
        // strip any folders → just filename
        String filename = resourceName.contains("/")
                ? resourceName.substring(resourceName.lastIndexOf('/') + 1)
                : resourceName;
        // load from CL/filename
        String src = KernelUtils.loadResourceAsString("CL/" + filename);

        cl_program program = clCreateProgramWithSource(context, 1,
                new String[]{src}, null, null);
        clBuildProgram(program, 0, null, null, null, null);
        return program;
    }
}