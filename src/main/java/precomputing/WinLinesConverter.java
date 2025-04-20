package precomputing;

import java.io.*;
import java.nio.charset.StandardCharsets;

public class WinLinesConverter {
    public static void main(String[] args) throws IOException {
        // adjust these paths if you like
        String input = "src/main/resources/precomputed/winLines.txt";
        String outUpper = "src/main/resources/precomputed/winLinesUpper.txt";
        String outLower = "src/main/resources/precomputed/winLinesLower.txt";

        try (BufferedReader br = new BufferedReader(new FileReader(input, StandardCharsets.US_ASCII));
             BufferedWriter bwU = new BufferedWriter(new FileWriter(outUpper, StandardCharsets.US_ASCII));
             BufferedWriter bwL = new BufferedWriter(new FileWriter(outLower, StandardCharsets.US_ASCII))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] tok = line.trim().split("\\s+");
                if (tok.length != 3) continue;  // skip malformed lines

                char[] up = new char[3], lo = new char[3];
                for (int i = 0; i < 3; i++) {
                    int idx = Integer.parseInt(tok[i]);
                    if (idx < 0 || idx > 26) {
                        throw new IllegalArgumentException("Index out of range: " + idx);
                    }
                    up[i] = (idx == 26 ? '.' : (char) ('A' + idx));
                    lo[i] = (idx == 26 ? ',' : (char) ('a' + idx));
                }
                // write "A B C" style
                bwU.write(up[0] + " " + up[1] + " " + up[2]);
                bwU.newLine();
                bwL.write(lo[0] + " " + lo[1] + " " + lo[2]);
                bwL.newLine();
            }
        }

        System.out.println("Wrote winLinesUpper.txt and winLinesLower.txt");
    }
}
