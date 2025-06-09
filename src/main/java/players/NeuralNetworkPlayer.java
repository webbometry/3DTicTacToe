package players;

import game.Board;
import game.Game;
import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

public class NeuralNetworkPlayer extends Player {
    private static final String WEIGHTS_DIR =
            "C:/Users/webbometric/Documents/GitHub/3DTicTacToe/src/main/resources/NeuralNetwork";

    // network parameters
    private double[][] fc1Weight;
    private double[]   fc1Bias;
    private double[][] fc2Weight;
    private double[]   fc2Bias;
    private double[][] fc3Weight;
    private double[]   fc3Bias;

    public NeuralNetworkPlayer(Board.Player symbol) {
        super(symbol);
        loadLatestWeights();
    }

    @Override
    public String getName() {
        return "NeuralNet";
    }

    @Override
    public void makeMove(Game game) {
        Board board = game.getBoard();
        String s = board.board;            // 27-char string: 'x','o',' '

        // build input vector: +1 for our stones, -1 for opponent, 0 for empty
        double[] input = new double[27];
        char me = symbol.toChar();
        for (int i = 0; i < 27; i++) {
            char c = s.charAt(i);
            if      (c == me)           input[i] =  1.0;
            else if (c == ' ')          input[i] =  0.0;
            else                        input[i] = -1.0;
        }

        // forward pass: 27 → 128 → 64 → 27
        double[] h1     = new double[128];
        for (int i = 0; i < 128; i++) {
            h1[i] = relu(dot(fc1Weight[i], input) + fc1Bias[i]);
        }

        double[] h2     = new double[64];
        for (int i = 0; i < 64; i++) {
            h2[i] = relu(dot(fc2Weight[i], h1) + fc2Bias[i]);
        }

        double[] logits = new double[27];
        for (int i = 0; i < 27; i++) {
            logits[i] = dot(fc3Weight[i], h2) + fc3Bias[i];
        }

        // mask illegal (occupied) moves and pick best
        double bestVal = Double.NEGATIVE_INFINITY;
        int    bestIdx = -1;
        for (int i = 0; i < 27; i++) {
            if (s.charAt(i) == ' ' && logits[i] > bestVal) {
                bestVal = logits[i];
                bestIdx = i;
            }
        }
        if (bestIdx < 0) return;  // no valid moves

        // convert 0..26 back to (x,y,z)
        int x =  bestIdx %  3;
        int y = (bestIdx /   3) % 3;
        int z =  bestIdx /   9;

        game.applyMove(x, y, z);
    }

    // ─── helpers ────────────────────────────────────────────────────────────────

    private double relu(double x) {
        return x > 0 ? x : 0;
    }

    private double dot(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    /**
     * Find the weights_N.json file with the largest N in WEIGHTS_DIR,
     * then parse fc1..fc3 weights and biases out of it.
     */
    private void loadLatestWeights() {
        File dir = new File(WEIGHTS_DIR);
        File[] files = dir.listFiles(new FilenameFilter() {
            public boolean accept(File d, String name) {
                return name.startsWith("weights_") && name.endsWith(".json");
            }
        });
        if (files == null || files.length == 0) {
            throw new IllegalStateException("No weight files in " + WEIGHTS_DIR);
        }

        // pick highest episode number
        File best = null;
        int  bestEp = -1;
        for (File f : files) {
            String n = f.getName().replace("weights_", "").replace(".json", "");
            try {
                int ep = Integer.parseInt(n);
                if (ep > bestEp) {
                    bestEp = ep;
                    best   = f;
                }
            } catch (NumberFormatException x) { /* skip */ }
        }
        if (best == null) {
            throw new IllegalStateException("No valid weight files in " + WEIGHTS_DIR);
        }

        // parse JSON
        try {
            ObjectMapper mapper = new ObjectMapper();
            JsonNode root = mapper.readTree(best);

            // fc1
            JsonNode n1w = root.get("fc1.weight");
            int out1 = n1w.size(), in1 = n1w.get(0).size();
            fc1Weight = new double[out1][in1];
            for (int i = 0; i < out1; i++)
                for (int j = 0; j < in1; j++)
                    fc1Weight[i][j] = n1w.get(i).get(j).asDouble();

            JsonNode n1b = root.get("fc1.bias");
            fc1Bias = new double[n1b.size()];
            for (int i = 0; i < n1b.size(); i++)
                fc1Bias[i] = n1b.get(i).asDouble();

            // fc2
            JsonNode n2w = root.get("fc2.weight");
            int out2 = n2w.size(), in2 = n2w.get(0).size();
            fc2Weight = new double[out2][in2];
            for (int i = 0; i < out2; i++)
                for (int j = 0; j < in2; j++)
                    fc2Weight[i][j] = n2w.get(i).get(j).asDouble();

            JsonNode n2b = root.get("fc2.bias");
            fc2Bias = new double[n2b.size()];
            for (int i = 0; i < n2b.size(); i++)
                fc2Bias[i] = n2b.get(i).asDouble();

            // fc3
            JsonNode n3w = root.get("fc3.weight");
            int out3 = n3w.size(), in3 = n3w.get(0).size();
            fc3Weight = new double[out3][in3];
            for (int i = 0; i < out3; i++)
                for (int j = 0; j < in3; j++)
                    fc3Weight[i][j] = n3w.get(i).get(j).asDouble();

            JsonNode n3b = root.get("fc3.bias");
            fc3Bias = new double[n3b.size()];
            for (int i = 0; i < n3b.size(); i++)
                fc3Bias[i] = n3b.get(i).asDouble();

        } catch (IOException e) {
            throw new RuntimeException("Failed to load weights from "
                    + best.getAbsolutePath(), e);
        }

        System.out.println("[NeuralNet] Loaded weights from " + best.getName());
    }
}
