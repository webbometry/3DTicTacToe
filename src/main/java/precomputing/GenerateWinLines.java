package precomputing;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

public class GenerateWinLines {

    /**
     * Converts 3D board coordinates to a string index.
     * The board is assumed to be 3x3x3, with indices computed as:
     * index = x + y * 3 + z * 9.
     */
    private static int toIndex(int x, int y, int z) {
        return x + y * 3 + z * 9;
    }

    public static void main(String[] args) {
        // The output file to write each winning line.
        String filename = "winLines.txt";

        try (PrintWriter writer = new PrintWriter(new FileWriter(filename))) {
            // Iterate through each cell of the 3x3x3 board.
            for (int z = 0; z < 3; z++) {
                for (int y = 0; y < 3; y++) {
                    for (int x = 0; x < 3; x++) {
                        // Iterate over possible direction vectors.
                        // dx, dy, dz can be -1, 0, or 1.
                        // We skip the zero vector and any vector that changes in all three axes.
                        for (int dz = -1; dz <= 1; dz++) {
                            for (int dy = -1; dy <= 1; dy++) {
                                for (int dx = -1; dx <= 1; dx++) {
                                    // Skip the zero vector.
                                    if (dx == 0 && dy == 0 && dz == 0) {
                                        continue;
                                    }
                                    // Exclude moves that span all three axes (not allowed by the custom rule).
                                    if (dx != 0 && dy != 0 && dz != 0) {
                                        continue;
                                    }
                                    // To avoid duplicate lines (counted in reverse order), we only use one “orientation.”
                                    // This condition makes sure that for any given vector, we only process it in one direction.
                                    if (dx < 0 || (dx == 0 && dy < 0) || (dx == 0 && dy == 0 && dz < 0)) {
                                        continue;
                                    }

                                    // Check if starting at (x,y,z) and taking two steps in direction (dx,dy,dz)
                                    // stays inside the board.
                                    int xEnd = x + 2 * dx;
                                    int yEnd = y + 2 * dy;
                                    int zEnd = z + 2 * dz;
                                    if (xEnd < 0 || xEnd >= 3 || yEnd < 0 || yEnd >= 3 || zEnd < 0 || zEnd >= 3) {
                                        continue;
                                    }

                                    // Compute the indexes for the three cells in the line.
                                    int index1 = toIndex(x, y, z);
                                    int index2 = toIndex(x + dx, y + dy, z + dz);
                                    int index3 = toIndex(x + 2 * dx, y + 2 * dy, z + 2 * dz);

                                    // Write the winning line indexes to file (space-delimited).
                                    writer.println(index1 + " " + index2 + " " + index3);
                                }
                            }
                        }
                    }
                }
            }
            System.out.println("Winning lines have been written to " + filename);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
