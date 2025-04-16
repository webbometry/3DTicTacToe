package game;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class Board {
    public String board;

    public Board() {
        board = newEmptyBoard();
    }

    public static int toIndex(int x, int y, int z) {
        return x + y * 3 + z * 9;
    }

    public enum Player {
        X('x'),
        O('o'),
        NONE(' ');

        private final char ch;

        private Player(char description) {
            this.ch = description;
        }

        public char toChar() {
            return ch;
        }
    }

    public static class Result {
        public final boolean gameOver;
        public final Player winner;

        public Result(boolean gameOver, Player winner) {
            this.gameOver = gameOver;
            this.winner = winner;
        }
    }

    public void play(int x, int y, int z, Player player) {
        if (player == Player.NONE)
            throw new IllegalArgumentException("There has to be a selected player to play on a tile.");

        board = setTile(x, y, z, player.toChar());
    }

    private String setTile(int x, int y, int z, char player) {
        int index = toIndex(x, y, z);
        char tile = board.charAt(index);
        if (tile != ' ') throw new IllegalArgumentException("Cannot play on an already occupied tile.");
        StringBuilder sb = new StringBuilder(board);
        sb.setCharAt(index, player);
        return sb.toString();
    }

    public static String newEmptyBoard() {
        StringBuilder sb = new StringBuilder();
        int layers = 3;
        int rows = 3;
        int cols = 3;

        for (int space = 0; space < layers * rows * cols; space++) {
            sb.append(" ");
        }
        return sb.toString();
    }

    public Result checkWin() {
        int xWinCount = 0;
        int oWinCount = 0;

        try (BufferedReader br = new BufferedReader(new FileReader("winLines.txt"))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] tokens = line.trim().split("\\s+");
                if (tokens.length != 3) continue; // Skip malformed lines.
                try {
                    int i1 = Integer.parseInt(tokens[0]);
                    int i2 = Integer.parseInt(tokens[1]);
                    int i3 = Integer.parseInt(tokens[2]);

                    char c1 = board.charAt(i1);
                    char c2 = board.charAt(i2);
                    char c3 = board.charAt(i3);

                    if (c1 != ' ' && c1 == c2 && c2 == c3) {
                        if (Character.toLowerCase(c1) == 'x') {
                            xWinCount++;
                        } else if (Character.toLowerCase(c1) == 'o') {
                            oWinCount++;
                        }
                    }
                } catch (NumberFormatException | IndexOutOfBoundsException e) {
                    continue; // Skip lines that can't be parsed correctly.
                }

                if (xWinCount >= 2) return new Result(true, Player.X);
                if (oWinCount >= 2) return new Result(true, Player.O);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        boolean boardFull = true;
        for (int i = 0; i < board.length(); i++) {
            if (board.charAt(i) == ' ') {
                boardFull = false;
                break;
            }
        }
        if (boardFull) return new Result(true, Player.NONE);

        return new Result(false, Player.NONE);
    }
}
