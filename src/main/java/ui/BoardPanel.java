package ui;

import game.*;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;

public class BoardPanel extends JPanel {
    private Game game;
    private boolean showScores = true; // toggle for showing score numbers

    // Layout parameters.
    private final int sideMargin = 20;   // left/right margin
    private final int topMargin = 20;    // top margin
    private final int bottomMargin = 20; // bottom margin
    private final int boardGap = 20;     // vertical gap between boards

    public BoardPanel(Game game) {
        this.game = game;
        // Set the background to light blue.
        setBackground(new Color(173, 216, 230));

        addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                if (!isEnabled()) return;
                int panelWidth = getWidth() - 2 * sideMargin;
                int panelHeight = getHeight() - topMargin - bottomMargin;
                int totalGap = boardGap * 2; // two gaps for three boards
                int boardHeight = (panelHeight - totalGap) / 3;
                int boardWidth = panelWidth;

                // Determine which level was clicked.
                int yRelative = e.getY() - topMargin;
                int level = yRelative / (boardHeight + boardGap);
                if (level < 0 || level > 2) return;

                // Calculate the y offset for the board.
                int levelYOffset = topMargin + level * (boardHeight + boardGap);
                int yInBoard = e.getY() - levelYOffset;

                // Divide the board into 3 rows and 3 columns.
                int cellWidth = boardWidth / 3;
                int cellHeight = boardHeight / 3;
                int col = (e.getX() - sideMargin) / cellWidth;
                int row = yInBoard / cellHeight;

                // Validate indices.
                if (col < 0 || col >= 3 || row < 0 || row >= 3) return;

                // In Board.java, play(x,y,z,player) uses x=column, y=row, and z=level.
                game.applyMove(col, row, level);
            }
        });
    }

    public void toggleScores() {
        showScores = !showScores;
        repaint();
    }

    @Override
    public void setEnabled(boolean enabled) {
        super.setEnabled(enabled);
        // optional: give a visual cue when disabled
        repaint();
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);

        int panelWidth = getWidth() - 2 * sideMargin;
        int panelHeight = getHeight() - topMargin - bottomMargin;
        int totalGap = boardGap * 2;
        int boardHeight = (panelHeight - totalGap) / 3;
        int boardWidth = panelWidth;

        // For each of the three boards (levels).
        for (int level = 0; level < 3; level++) {
            int yOffset = topMargin + level * (boardHeight + boardGap);

            // Fill the board area with light blue (matching the main window).
            g.setColor(new Color(173, 216, 230));
            g.fillRect(sideMargin, yOffset, boardWidth, boardHeight);

            // Draw grid lines in grey.
            g.setColor(Color.BLUE);
            int cellWidth = boardWidth / 3;
            int cellHeight = boardHeight / 3;
            for (int i = 0; i <= 3; i++) {
                int x = sideMargin + i * cellWidth;
                g.drawLine(x, yOffset, x, yOffset + boardHeight);
            }
            for (int j = 0; j <= 3; j++) {
                int y = yOffset + j * cellHeight;
                g.drawLine(sideMargin, y, sideMargin + boardWidth, y);
            }

            // Draw marks and scores.
            for (int row = 0; row < 3; row++) {
                for (int col = 0; col < 3; col++) {
                    int index = Board.toIndex(col, row, level);
                    char symbol = game.getBoard().board.charAt(index);
                    if (symbol != ' ') {
                        // Draw the player's mark.
                        if (Character.toLowerCase(symbol) == 'x') {
                            g.setColor(Color.BLACK);
                        } else if (Character.toLowerCase(symbol) == 'o') {
                            g.setColor(Color.WHITE);
                        }
                        g.setFont(new Font("SansSerif", Font.BOLD, 36));
                        int xPos = sideMargin + col * cellWidth + cellWidth / 2 - 10;
                        int yPos = yOffset + row * cellHeight + cellHeight / 2 + 10;
                        g.drawString(String.valueOf(symbol), xPos, yPos);
                    } else if (showScores) {
                        // Draw a default score "0". Scale the font with the cell size.
                        int scoreFontSize = Math.min(cellWidth, cellHeight) / 4;
                        g.setFont(new Font("SansSerif", Font.PLAIN, scoreFontSize));
                        g.setColor(Color.BLUE);
                        int xPos = sideMargin + col * cellWidth + cellWidth - (scoreFontSize + 5);
                        int yPos = yOffset + row * cellHeight + cellHeight - 5;
                        g.drawString("0", xPos, yPos);
                    }
                }
            }

            // Optionally, draw a small label for the level.
            g.setFont(new Font("SansSerif", Font.PLAIN, 12));
            g.setColor(Color.BLACK);
            g.drawString("Level " + level, sideMargin + 5, yOffset + 15);
        }

    }
}