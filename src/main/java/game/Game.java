package game;

import players.*;
import ui.*;

import javax.swing.*;
import java.awt.*;

public class Game extends JFrame {
    private Board board;
    private Player playerX;
    private Player playerO;
    public Player currentPlayer;
    private BoardPanel boardPanel;
    private EvaluationPanel evalPanel;
    private OptionsPanel optionsPanel;

    public Game() {
        board = new Board();
        playerX = new HumanPlayer(Board.Player.X);
        playerO = new HumanPlayer(Board.Player.O);
        currentPlayer = playerX;

        initUI();
    }

    private void initUI() {
        setTitle("3D Tic Tac Toe");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(900, 650);
        setLayout(new BorderLayout());

        // Set the main window background to light blue.
        getContentPane().setBackground(new Color(173, 216, 230));

        // Create and add the evaluation panel (left side).
        evalPanel = new EvaluationPanel();
        add(evalPanel, BorderLayout.WEST);

        // Create and add the board panel (center).
        boardPanel = new BoardPanel(this);
        add(boardPanel, BorderLayout.CENTER);

        // Create and add the options panel (right side).
        optionsPanel = new OptionsPanel(this, boardPanel, evalPanel);
        add(optionsPanel, BorderLayout.EAST);

        setLocationRelativeTo(null);
        setVisible(true);
    }

    public Board getBoard() {
        return board;
    }

    // x: column, y: row, z: level.
    public void applyMove(int x, int y, int z) {
        try {
            board.play(x, y, z, currentPlayer.getSymbol());
        } catch (IllegalArgumentException e) {
            return;
        }

        boardPanel.repaint();

        Board.Result result = board.checkWin();
        if (result.gameOver) {
            if (result.winner == Board.Player.NONE) {
                JOptionPane.showMessageDialog(this, "Draw!");
            } else {
                JOptionPane.showMessageDialog(this, "Player " + result.winner + " wins!");
            }
            board = new Board();
            boardPanel.repaint();
            currentPlayer = playerX;
            return;
        }

        currentPlayer = (currentPlayer == playerX) ? playerO : playerX;
        optionsPanel.updateCurrentMoveLabel();
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> new Game());
    }
}