package ui;

import game.*;

import javax.swing.*;
import java.awt.*;

public class OptionsPanel extends JPanel {
    private JCheckBox toggleScoresCheckBox;
    private JCheckBox toggleEvalBarCheckBox;
    private JComboBox<String> player1ComboBox;
    private JComboBox<String> player2ComboBox;

    // For now, only one option is available.
    private String[] playerOptions = {"Human"};

    private JLabel currentMoveLabel;

    private Game game;
    private BoardPanel boardPanel;
    private EvaluationPanel evalPanel;

    public OptionsPanel(Game game, BoardPanel boardPanel, EvaluationPanel evalPanel) {
        this.game = game;
        this.boardPanel = boardPanel;
        this.evalPanel = evalPanel;

        setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
        setPreferredSize(new Dimension(150, 600));

        toggleScoresCheckBox = new JCheckBox("Show Scores", true);
        toggleEvalBarCheckBox = new JCheckBox("Show Eval Bar", true);

        // Set dropdowns to a smaller height.
        player1ComboBox = new JComboBox<>(playerOptions);
        player1ComboBox.setMaximumSize(new Dimension(120, 25));
        player2ComboBox = new JComboBox<>(playerOptions);
        player2ComboBox.setMaximumSize(new Dimension(120, 25));

        JLabel player1Label = new JLabel("Player 1:");
        JLabel player2Label = new JLabel("Player 2:");

        toggleScoresCheckBox.addActionListener(e -> boardPanel.toggleScores());
        toggleEvalBarCheckBox.addActionListener(e -> evalPanel.setVisible(toggleEvalBarCheckBox.isSelected()));

        // Log the player selection for now.
        player1ComboBox.addActionListener(e ->
                System.out.println("Player 1 set to: " + player1ComboBox.getSelectedItem()));
        player2ComboBox.addActionListener(e ->
                System.out.println("Player 2 set to: " + player2ComboBox.getSelectedItem()));

        add(Box.createRigidArea(new Dimension(0, 20)));
        add(toggleScoresCheckBox);
        add(Box.createRigidArea(new Dimension(0, 20)));
        add(toggleEvalBarCheckBox);
        add(Box.createRigidArea(new Dimension(0, 40)));
        add(player1Label);
        add(player1ComboBox);
        add(Box.createRigidArea(new Dimension(0, 20)));
        add(player2Label);
        add(player2ComboBox);

        // Add a new label at the bottom to display the current move.
        currentMoveLabel = new JLabel("Current Move: " + game.currentPlayer.getSymbol());
        System.out.println("update");
        add(Box.createRigidArea(new Dimension(0, 20)));
        add(currentMoveLabel);
    }
    public void updateCurrentMoveLabel() {
        SwingUtilities.invokeLater(() -> {
            if (currentMoveLabel != null && game != null && game.currentPlayer != null) {
                currentMoveLabel.setText("Current Move: " + game.currentPlayer.getSymbol());
                currentMoveLabel.revalidate();
                currentMoveLabel.repaint();
            }
        });
    }
}
