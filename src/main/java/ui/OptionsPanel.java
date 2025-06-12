// src/main/java/ui/OptionsPanel.java
package ui;

import game.Board;
import game.Game;
import players.HumanPlayer;
import players.NeuralNetworkPlayer;

import javax.swing.*;
import java.awt.*;

public class OptionsPanel extends JPanel {
    private final Game game;
    private final BoardPanel boardPanel;
    private final EvaluationPanel evalPanel;

    private final JCheckBox toggleScoresCheckBox;
    private final JCheckBox toggleEvalBarCheckBox;
    private final JComboBox<String> player1ComboBox;
    private final JComboBox<String> player2ComboBox;
    private final JButton nextButton;
    private final JLabel currentMoveLabel;

    private final String[] playerOptions = { "Human", "NeuralNet" };

    public OptionsPanel(Game game, BoardPanel boardPanel, EvaluationPanel evalPanel) {
        this.game       = game;
        this.boardPanel = boardPanel;
        this.evalPanel  = evalPanel;

        setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
        setPreferredSize(new Dimension(150, 600));

        // ─── Score & Eval toggles ────────────────────────────────────────
        toggleScoresCheckBox = new JCheckBox("Show Scores", false);
        toggleEvalBarCheckBox = new JCheckBox("Show Eval Bar", false);
        toggleScoresCheckBox.addActionListener(e -> boardPanel.toggleScores());
        toggleEvalBarCheckBox.addActionListener(e ->
                evalPanel.setVisible(toggleEvalBarCheckBox.isSelected()));
        add(Box.createVerticalStrut(10));
        add(toggleScoresCheckBox);
        add(Box.createVerticalStrut(10));
        add(toggleEvalBarCheckBox);
        evalPanel.setVisible(false);
        add(Box.createVerticalStrut(20));

        // ─── Player X selector ──────────────────────────────────────────
        add(new JLabel("Player X:"));
        player1ComboBox = new JComboBox<>(playerOptions);
        player1ComboBox.setMaximumSize(new Dimension(120, 25));
        player1ComboBox.addActionListener(e -> {
            String sel = (String) player1ComboBox.getSelectedItem();
            if ("Human".equals(sel)) {
                game.setPlayerX(new HumanPlayer(Board.Player.X));
            } else {
                game.setPlayerX(new NeuralNetworkPlayer(Board.Player.X));
            }
            updateCurrentMoveLabel();
        });
        add(player1ComboBox);
        add(Box.createVerticalStrut(20));

        // ─── Player O selector ──────────────────────────────────────────
        add(new JLabel("Player O:"));
        player2ComboBox = new JComboBox<>(playerOptions);
        player2ComboBox.setMaximumSize(new Dimension(120, 25));
        player2ComboBox.addActionListener(e -> {
            String sel = (String) player2ComboBox.getSelectedItem();
            if ("Human".equals(sel)) {
                game.setPlayerO(new HumanPlayer(Board.Player.O));
            } else {
                game.setPlayerO(new NeuralNetworkPlayer(Board.Player.O));
            }
            updateCurrentMoveLabel();
        });
        add(player2ComboBox);
        add(Box.createVerticalStrut(20));

        // ─── Current‐Move Display ────────────────────────────────────────
        currentMoveLabel = new JLabel("Current Move: " + game.currentPlayer.getSymbol());
        add(currentMoveLabel);
        add(Box.createVerticalStrut(20));

        // ─── Next Move button ───────────────────────────────────────────
        add(new JLabel("AI Control:"));
        nextButton = new JButton("Next Move");
        nextButton.setAlignmentX(Component.CENTER_ALIGNMENT);
        nextButton.setMaximumSize(new Dimension(120, 30));
        nextButton.setEnabled(false);
        nextButton.addActionListener(e -> game.stepAIMove());
        add(nextButton);

        // initialize button state
        updateControlButtons();
    }

    /** Refresh the “Current Move” label. */
    public void updateCurrentMoveLabel() {
        currentMoveLabel.setText("Current Move: " + game.currentPlayer.getSymbol());
    }

    /**
     * Enable Next Move if:
     *  • both players are AI, or
     *  • it’s the very first move and the current player is an AI.
     */
    public void updateControlButtons() {
        boolean bothAI = !(game.getPlayerX() instanceof HumanPlayer)
                && !(game.getPlayerO() instanceof HumanPlayer);
        boolean firstAINow = !game.isFirstMoveDone()
                && (game.currentPlayer instanceof NeuralNetworkPlayer);
        nextButton.setEnabled(bothAI || firstAINow);
    }
}
