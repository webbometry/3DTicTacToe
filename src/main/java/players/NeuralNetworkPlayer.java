package players;

import game.Board;
import game.Game;
import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.core.type.TypeReference;

public class NeuralNetworkPlayer extends Player {
    private static final String WEIGHTS_DIR =
            "C:/Users/webbometric/Documents/GitHub/3DTicTacToe/src/main/resources/NeuralNetwork";

    // ---- model parameters matching PolicyValueNet: conv1, conv2, conv3, fc_common, fc_policy ----
    private double[][][][][] conv1Weight;   // [64][1][3][3][3]
    private double[]           conv1Bias;    // [64]

    private double[][][][][] conv2Weight;   // [64][64][3][3][3]
    private double[]           conv2Bias;    // [64]

    private double[][][][][] conv3Weight;   // [64][64][3][3][3]
    private double[]           conv3Bias;    // [64]

    private double[][] fcCommonWeight;      // [256][64*3*3*3]
    private double[]   fcCommonBias;        // [256]

    private double[][] fcPolicyWeight;      // [27][256]
    private double[]   fcPolicyBias;        // [27]

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
        String s = board.board; // 27-char string

        // --- build input tensor [1][3][3][3] ---
        double[][][][] input = new double[1][3][3][3];
        char me = symbol.toChar();
        for (int i = 0; i < 27; i++) {
            char c = s.charAt(i);
            double v = (c == me ? 1.0 : c == ' ' ? 0.0 : -1.0);
            int x =   i % 3;
            int y = ( i / 3) % 3;
            int z =   i / 9;
            input[0][x][y][z] = v;
        }

        // --- conv1 (64→1→3×3×3, padding=1) + ReLU → out1[64][3][3][3] ---
        double[][][][] out1 = new double[64][3][3][3];
        for (int oc = 0; oc < 64; oc++) {
            for (int x = 0; x < 3; x++) for (int y = 0; y < 3; y++) for (int z = 0; z < 3; z++) {
                double sum = conv1Bias[oc];
                for (int ic = 0; ic < 1; ic++)
                    for (int dx = -1; dx <= 1; dx++)
                        for (int dy = -1; dy <= 1; dy++)
                            for (int dz = -1; dz <= 1; dz++) {
                                int xx = x + dx, yy = y + dy, zz = z + dz;
                                double vv = (xx>=0 && xx<3 && yy>=0 && yy<3 && zz>=0 && zz<3)
                                        ? input[ic][xx][yy][zz] : 0.0;
                                sum += conv1Weight[oc][ic][dx+1][dy+1][dz+1] * vv;
                            }
                out1[oc][x][y][z] = relu(sum);
            }
        }

        // --- conv2 (64→64→3×3×3, pad=1) + ReLU → out2[64][3][3][3] ---
        double[][][][] out2 = new double[64][3][3][3];
        for (int oc = 0; oc < 64; oc++) {
            for (int x = 0; x < 3; x++) for (int y = 0; y < 3; y++) for (int z = 0; z < 3; z++) {
                double sum = conv2Bias[oc];
                for (int ic = 0; ic < 64; ic++)
                    for (int dx = -1; dx <= 1; dx++)
                        for (int dy = -1; dy <= 1; dy++)
                            for (int dz = -1; dz <= 1; dz++) {
                                int xx = x + dx, yy = y + dy, zz = z + dz;
                                double vv = (xx>=0 && xx<3 && yy>=0 && yy<3 && zz>=0 && zz<3)
                                        ? out1[ic][xx][yy][zz] : 0.0;
                                sum += conv2Weight[oc][ic][dx+1][dy+1][dz+1] * vv;
                            }
                out2[oc][x][y][z] = relu(sum);
            }
        }

        // --- conv3 (64→64→3×3×3, pad=1) + ReLU → out3[64][3][3][3] ---
        double[][][][] out3 = new double[64][3][3][3];
        for (int oc = 0; oc < 64; oc++) {
            for (int x = 0; x < 3; x++) for (int y = 0; y < 3; y++) for (int z = 0; z < 3; z++) {
                double sum = conv3Bias[oc];
                for (int ic = 0; ic < 64; ic++)
                    for (int dx = -1; dx <= 1; dx++)
                        for (int dy = -1; dy <= 1; dy++)
                            for (int dz = -1; dz <= 1; dz++) {
                                int xx = x + dx, yy = y + dy, zz = z + dz;
                                double vv = (xx>=0 && xx<3 && yy>=0 && yy<3 && zz>=0 && zz<3)
                                        ? out2[ic][xx][yy][zz] : 0.0;
                                sum += conv3Weight[oc][ic][dx+1][dy+1][dz+1] * vv;
                            }
                out3[oc][x][y][z] = relu(sum);
            }
        }

        // --- flatten out3 → flat[64*3*3*3] ---
        double[] flat = new double[64*3*3*3];
        int idx = 0;
        for (int oc = 0; oc < 64; oc++)
            for (int x = 0; x < 3; x++)
                for (int y = 0; y < 3; y++)
                    for (int z = 0; z < 3; z++)
                        flat[idx++] = out3[oc][x][y][z];

        // --- fc_common: 64*3*3*3→256 + ReLU →
        double[] common = new double[256];
        for (int i = 0; i < 256; i++) {
            double sum = fcCommonBias[i];
            for (int j = 0; j < flat.length; j++)
                sum += fcCommonWeight[i][j] * flat[j];
            common[i] = relu(sum);
        }

        // --- fc_policy: 256→27 (no activation) → logits[27] ---
        double[] logits = new double[27];
        for (int i = 0; i < 27; i++) {
            double sum = fcPolicyBias[i];
            for (int j = 0; j < 256; j++)
                sum += fcPolicyWeight[i][j] * common[j];
            logits[i] = sum;
        }

        // --- pick highest‐logit legal move ---
        double bestVal = Double.NEGATIVE_INFINITY;
        int    bestIdx = -1;
        for (int i = 0; i < 27; i++) {
            if (s.charAt(i) == ' ' && logits[i] > bestVal) {
                bestVal = logits[i];
                bestIdx = i;
            }
        }
        if (bestIdx < 0) return;
        int x =   bestIdx % 3;
        int y = ( bestIdx / 3) % 3;
        int z =   bestIdx / 9;
        game.applyMove(x, y, z);
    }

    private double relu(double x) {
        return x > 0 ? x : 0;
    }

    private void loadLatestWeights() {
        File dir = new File(WEIGHTS_DIR);
        File[] files = dir.listFiles((d, n) -> n.matches("weights_\\d+\\.json"));
        if (files == null || files.length == 0)
            throw new RuntimeException("No weight JSON files found in " + WEIGHTS_DIR);

        // pick the highest-numbered file
        File latest = null;
        int maxNum  = -1;
        for (File f : files) {
            int num = Integer.parseInt(f.getName().replaceAll("\\D", ""));
            if (num > maxNum) { maxNum = num; latest = f; }
        }

        ObjectMapper mapper = new ObjectMapper();
        try {
            JsonNode root = mapper.readTree(latest);

            conv1Weight    = mapper.convertValue(root.get("conv1.weight"),
                    new TypeReference<double[][][][][]>(){});
            conv1Bias      = mapper.convertValue(root.get("conv1.bias"),
                    new TypeReference<double[]>(){});

            conv2Weight    = mapper.convertValue(root.get("conv2.weight"),
                    new TypeReference<double[][][][][]>(){});
            conv2Bias      = mapper.convertValue(root.get("conv2.bias"),
                    new TypeReference<double[]>(){});

            conv3Weight    = mapper.convertValue(root.get("conv3.weight"),
                    new TypeReference<double[][][][][]>(){});
            conv3Bias      = mapper.convertValue(root.get("conv3.bias"),
                    new TypeReference<double[]>(){});

            fcCommonWeight = mapper.convertValue(root.get("fc_common.weight"),
                    new TypeReference<double[][]>(){});
            fcCommonBias   = mapper.convertValue(root.get("fc_common.bias"),
                    new TypeReference<double[]>(){});

            fcPolicyWeight = mapper.convertValue(root.get("fc_policy.weight"),
                    new TypeReference<double[][]>(){});
            fcPolicyBias   = mapper.convertValue(root.get("fc_policy.bias"),
                    new TypeReference<double[]>(){});
        } catch (IOException ex) {
            throw new RuntimeException("Failed to load neural network weights", ex);
        }
    }
}
