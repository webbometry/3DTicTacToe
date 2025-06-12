// src/main/java/game/Game.java
package game;

import players.HumanPlayer;
import players.Player;
import ui.BoardPanel;
import ui.EvaluationPanel;
import ui.OptionsPanel;

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
    private boolean firstMoveDone = false;

    public Game() {
        board         = new Board();
        playerX       = new HumanPlayer(Board.Player.X);
        playerO       = new HumanPlayer(Board.Player.O);
        currentPlayer = playerX;

        initUI();
        updateTurnControls();  // decide if we auto‐step or wait for Next Move
    }

    private void initUI() {
        setTitle("3D Tic Tac Toe");
        setDefaultCloseOperation(EXIT_ON_CLOSE);
        setSize(900, 650);
        setLayout(new BorderLayout());
        getContentPane().setBackground(new Color(173, 216, 230));

        evalPanel    = new EvaluationPanel();
        boardPanel   = new BoardPanel(this);
        optionsPanel = new OptionsPanel(this, boardPanel, evalPanel);

        add(evalPanel,    BorderLayout.WEST);
        add(boardPanel,   BorderLayout.CENTER);
        add(optionsPanel, BorderLayout.EAST);

        setLocationRelativeTo(null);
        setVisible(true);
    }

    public Board getBoard()       { return board; }
    public Player getPlayerX()    { return playerX; }
    public Player getPlayerO()    { return playerO; }
    public boolean isFirstMoveDone() { return firstMoveDone; }

    /**
     * Called by both human clicks and AI moves.
     */
    public void applyMove(int x, int y, int z) {
        boolean wasFirst = !firstMoveDone;
        try {
            board.play(x, y, z, currentPlayer.getSymbol());
        } catch (IllegalArgumentException e) {
            return;  // invalid move
        }
        if (wasFirst) firstMoveDone = true;

        boardPanel.repaint();
        var result = board.checkWin();
        if (result.gameOver) {
            String msg = result.winner == Board.Player.NONE
                    ? "Draw!"
                    : "Player " + result.winner + " wins!";
            JOptionPane.showMessageDialog(this, msg);

            // reset everything
            board         = new Board();
            boardPanel.repaint();
            currentPlayer = playerX;
            firstMoveDone = false;
            optionsPanel.updateCurrentMoveLabel();
            updateTurnControls();
            return;
        }

        // next turn
        currentPlayer = (currentPlayer == playerX ? playerO : playerX);
        optionsPanel.updateCurrentMoveLabel();
        updateTurnControls();
    }

    /** Invoked by the Next Move button. */
    public void stepAIMove() {
        currentPlayer.makeMove(this);
    }

    /**
     * Decides whether to auto‐step the AI or wait for Next Move.
     */
    private void updateTurnControls() {
        boolean humanTurn = currentPlayer instanceof HumanPlayer;
        boolean bothAI    = !(playerX instanceof HumanPlayer)
                && !(playerO instanceof HumanPlayer);

        // human may click only on human turns
        boardPanel.setEnabled(humanTurn);

        // enable Next Move if either:
        //  • both players are AI, or
        //  • it's the very first move and currentPlayer is AI
        optionsPanel.updateControlButtons();

        // If this is *not* the first move, *and* it's AI vs Human,
        // auto‐step the AI immediately.
        if (!humanTurn && !bothAI && firstMoveDone) {
            SwingUtilities.invokeLater(() -> currentPlayer.makeMove(this));
        }
    }

    public void setPlayerX(Player p) {
        this.playerX = p;
        if (currentPlayer.getSymbol() == Board.Player.X) {
            currentPlayer = p;
            optionsPanel.updateCurrentMoveLabel();
        }
        updateTurnControls();
    }

    public void setPlayerO(Player p) {
        this.playerO = p;
        if (currentPlayer.getSymbol() == Board.Player.O) {
            currentPlayer = p;
            optionsPanel.updateCurrentMoveLabel();
        }
        updateTurnControls();
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(Game::new);
    }
}
