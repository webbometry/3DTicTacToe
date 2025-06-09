// src/main/java/game/Game.java
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
        board      = new Board();
        playerX    = new HumanPlayer(Board.Player.X);
        playerO    = new HumanPlayer(Board.Player.O);
        currentPlayer = playerX;

        initUI();

        // At startup: enable board if human, or fire the NN immediately
        boolean humanTurn = currentPlayer instanceof HumanPlayer;
        boardPanel.setEnabled(humanTurn);
        if (!humanTurn) {
            SwingUtilities.invokeLater(() -> currentPlayer.makeMove(this));
        }
    }

    private void initUI() {
        setTitle("3D Tic Tac Toe");
        setDefaultCloseOperation(EXIT_ON_CLOSE);
        setSize(900, 650);
        setLayout(new BorderLayout());
        getContentPane().setBackground(new Color(173, 216, 230));

        evalPanel   = new EvaluationPanel();
        boardPanel  = new BoardPanel(this);
        optionsPanel= new OptionsPanel(this, boardPanel, evalPanel);

        add(evalPanel,    BorderLayout.WEST);
        add(boardPanel,   BorderLayout.CENTER);
        add(optionsPanel, BorderLayout.EAST);

        setLocationRelativeTo(null);
        setVisible(true);
    }

    public Board getBoard() {
        return board;
    }

    /**
     * Called by both human clicks and NN moves.
     */
    public void applyMove(int x, int y, int z) {
        try {
            board.play(x, y, z, currentPlayer.getSymbol());
        } catch (IllegalArgumentException e) {
            // invalid click—ignore
            return;
        }

        boardPanel.repaint();

        // Check for end-of-game
        Board.Result result = board.checkWin();
        if (result.gameOver) {
            if (result.winner == Board.Player.NONE) {
                JOptionPane.showMessageDialog(this, "Draw!");
            } else {
                JOptionPane.showMessageDialog(this,
                        "Player " + result.winner + " wins!");
            }

            // reset
            board = new Board();
            boardPanel.repaint();
            currentPlayer = playerX;
            optionsPanel.updateCurrentMoveLabel();
            setPlayerX(new HumanPlayer(Board.Player.X));
            setPlayerO(new HumanPlayer(Board.Player.O));

            // re-enable/NN-play
            boolean humanTurn2 = currentPlayer instanceof HumanPlayer;
            boardPanel.setEnabled(humanTurn2);
            if (!humanTurn2) {
                SwingUtilities.invokeLater(() -> currentPlayer.makeMove(this));
            }
            return;
        }

        // switch sides
        currentPlayer = (currentPlayer == playerX) ? playerO : playerX;
        optionsPanel.updateCurrentMoveLabel();

        // enable board only for human; queue NN otherwise
        boolean humanTurn2 = currentPlayer instanceof HumanPlayer;
        boardPanel.setEnabled(humanTurn2);
        if (!humanTurn2) {
            SwingUtilities.invokeLater(() -> currentPlayer.makeMove(this));
        }
    }

    /** Swap in a new X-player and, if it’s X’s turn, trigger it immediately. */
    public void setPlayerX(Player p) {
        this.playerX = p;
        if (currentPlayer.getSymbol() == Board.Player.X) {
            currentPlayer = p;
            optionsPanel.updateCurrentMoveLabel();
            boolean human = p instanceof HumanPlayer;
            boardPanel.setEnabled(human);
            if (!human) SwingUtilities.invokeLater(() -> p.makeMove(this));
        }
    }

    /** Swap in a new O-player and, if it’s O’s turn, trigger it immediately. */
    public void setPlayerO(Player p) {
        this.playerO = p;
        if (currentPlayer.getSymbol() == Board.Player.O) {
            currentPlayer = p;
            optionsPanel.updateCurrentMoveLabel();
            boolean human = p instanceof HumanPlayer;
            boardPanel.setEnabled(human);
            if (!human) SwingUtilities.invokeLater(() -> p.makeMove(this));
        }
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> new Game());
    }
}
