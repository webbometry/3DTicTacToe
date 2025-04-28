package game;

public class Minimax {
    public String convert(String board) {
        StringBuilder sb = new StringBuilder();
        for (int pos = 0; pos < board.length(); pos++) {
            if (board.charAt(pos) == 'x') {
                if (pos == 26) sb.append('.');
                else sb.append('A' + pos);
            } else if (board.charAt(pos) == 'o') {
                if (pos == 26) sb.append(',');
                else sb.append('a' + pos);
            }
        }
        return sb.toString();
    }
}
