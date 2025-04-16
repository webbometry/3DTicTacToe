package precomputing;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * GenerateRotationMaps computes the 24 proper rotation mappings for a 3x3x3 board.
 * Each mapping is represented as an int[27] array (mapping[i] is the new index for cell i).
 * The mappings are saved to a file (rotationMaps.txt), one mapping per line with 27 integers.
 */
public class GenerateRotationMaps {

    public static void main(String[] args) {
        // Generate the rotation maps.
        List<int[]> rotationMaps = generateRotationMaps();
        System.out.println("Total rotation maps: " + rotationMaps.size());

        // Save to file; adjust the filename/path as needed.
        String filename = "rotationMaps.txt";
        try {
            saveRotationMapsToFile(rotationMaps, filename);
            System.out.println("Rotation maps saved to " + filename);
        } catch (IOException e) {
            System.err.println("Error saving rotation maps: " + e.getMessage());
        }
    }

    /**
     * Computes the 24 rotation mappings.
     * For each rotation matrix, computes the new board index for every cell.
     *
     * @return a List of int arrays, each array is a mapping for one rotation.
     */
    private static List<int[]> generateRotationMaps() {
        List<int[]> maps = new ArrayList<>();
        List<int[][]> matrices = generateRotationMatrices();
        int boardSize = 27;

        for (int[][] matrix : matrices) {
            int[] mapping = new int[boardSize];
            // For each cell (index 0 to 26), compute its new index under this rotation.
            for (int index = 0; index < boardSize; index++) {
                int x = index % 3;
                int y = (index / 3) % 3;
                int z = index / 9;
                // Shift to center coordinates (center = (1,1,1)).
                int cx = x - 1;
                int cy = y - 1;
                int cz = z - 1;
                // Apply the rotation matrix.
                int nx = matrix[0][0] * cx + matrix[0][1] * cy + matrix[0][2] * cz;
                int ny = matrix[1][0] * cx + matrix[1][1] * cy + matrix[1][2] * cz;
                int nz = matrix[2][0] * cx + matrix[2][1] * cy + matrix[2][2] * cz;
                // Shift back from center coordinates.
                int newX = nx + 1;
                int newY = ny + 1;
                int newZ = nz + 1;
                mapping[index] = newX + newY * 3 + newZ * 9;
            }
            maps.add(mapping);
        }

        return maps;
    }

    /**
     * Generates all 24 proper rotation matrices (3x3 integer arrays) for a cube.
     * Based on choosing a valid "up" vector and a perpendicular "front" vector,
     * then computing rotations (0, 90, 180, 270 degrees) about the up-axis.
     *
     * @return a List of rotation matrices.
     */
    private static List<int[][]> generateRotationMatrices() {
        List<int[][]> matrices = new ArrayList<>();
        // Candidate "up" unit vectors.
        int[][] ups = {
                {0, 0, 1},
                {0, 0, -1},
                {0, 1, 0},
                {0, -1, 0},
                {1, 0, 0},
                {-1, 0, 0}
        };

        for (int[] up : ups) {
            List<int[]> fronts = new ArrayList<>();
            int[][] candidates = {
                    {0, 1, 0},
                    {0, -1, 0},
                    {1, 0, 0},
                    {-1, 0, 0},
                    {0, 0, 1},
                    {0, 0, -1}
            };
            // Choose candidate front vectors that are perpendicular to up.
            for (int[] cand : candidates) {
                if (dot(up, cand) == 0) {
                    fronts.add(cand);
                }
            }
            // For each valid front, generate rotation matrices by rotating about the up-axis.
            for (int[] front : fronts) {
                int[] right = cross(up, front);
                for (int theta : new int[]{0, 90, 180, 270}) {
                    double rad = Math.toRadians(theta);
                    int[] rFront = new int[3];
                    int[] rRight = new int[3];
                    for (int i = 0; i < 3; i++) {
                        // Rotate front vector.
                        rFront[i] = (int) Math.round(Math.cos(rad) * front[i] + Math.sin(rad) * right[i]);
                        // Rotate right vector.
                        rRight[i] = (int) Math.round(-Math.sin(rad) * front[i] + Math.cos(rad) * right[i]);
                    }
                    // Build the rotation matrix. Columns: rRight, rFront, up.
                    int[][] mat = new int[3][3];
                    // First column: right vector.
                    mat[0][0] = rRight[0];
                    mat[1][0] = rRight[1];
                    mat[2][0] = rRight[2];
                    // Second column: front vector.
                    mat[0][1] = rFront[0];
                    mat[1][1] = rFront[1];
                    mat[2][1] = rFront[2];
                    // Third column: up vector.
                    mat[0][2] = up[0];
                    mat[1][2] = up[1];
                    mat[2][2] = up[2];

                    if (!containsMatrix(matrices, mat)) {
                        matrices.add(mat);
                    }
                }
            }
        }
        return matrices;
    }

    /**
     * Computes the cross product of two 3D vectors.
     */
    private static int[] cross(int[] a, int[] b) {
        return new int[]{
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0]
        };
    }

    /**
     * Computes the dot product of two 3D vectors.
     */
    private static int dot(int[] a, int[] b) {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    }

    /**
     * Checks whether the provided list of matrices already contains an equivalent matrix.
     */
    private static boolean containsMatrix(List<int[][]> matrices, int[][] mat) {
        for (int[][] m : matrices) {
            if (areMatricesEqual(m, mat))
                return true;
        }
        return false;
    }

    /**
     * Compares two 3x3 matrices for equality.
     */
    private static boolean areMatricesEqual(int[][] a, int[][] b) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (a[i][j] != b[i][j])
                    return false;
            }
        }
        return true;
    }

    /**
     * Saves the list of rotation maps to a file.
     * Each line in the file represents one mapping: 27 integers separated by spaces.
     *
     * @param maps     the list of rotation maps.
     * @param filename the name of the file to write.
     * @throws IOException if an I/O error occurs.
     */
    private static void saveRotationMapsToFile(List<int[]> maps, String filename) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filename))) {
            for (int[] map : maps) {
                for (int i = 0; i < map.length; i++) {
                    writer.write(Integer.toString(map[i]));
                    if (i < map.length - 1) {
                        writer.write(" ");
                    }
                }
                writer.newLine();
            }
        }
    }
}
