package precomputing.minimax;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.stream.Collectors;

public class KernelUtils {
    /**
     * Load a resource text file from the classpath.
     * Tries the exact path, then the classloader root, then under CL/.
     */
    public static String loadResourceAsString(String path) {
        String[] candidates = new String[]{
                path,
                path.startsWith("/") ? path.substring(1) : "/" + path,
                // if path had folders, try just the filename
                path.contains("/") ? path.substring(path.lastIndexOf('/') + 1) : path,
                // and under CL/
                "CL/" + (path.contains("/") ? path.substring(path.lastIndexOf('/') + 1) : path)
        };

        InputStream in = null;
        for (String p : candidates) {
            in = KernelUtils.class.getResourceAsStream(p);
            if (in == null) {
                in = KernelUtils.class.getClassLoader().getResourceAsStream(p);
            }
            if (in != null) {
                try (BufferedReader reader = new BufferedReader(new InputStreamReader(in))) {
                    return reader.lines().collect(Collectors.joining("\n"));
                } catch (Exception e) {
                    throw new RuntimeException("Error reading resource: " + p, e);
                }
            }
        }
        throw new IllegalArgumentException("Resource not found (tried): " + String.join(", ", candidates));
    }
}