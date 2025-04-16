package ui;

import javax.swing.*;
import java.awt.*;

public class EvaluationPanel extends JPanel {
    // Evaluation value (range: -27 to 27). For now default is 0.
    private int evaluation = 0;

    public EvaluationPanel() {
        setPreferredSize(new Dimension(80, 600));
    }

    public void setEvaluation(int eval) {
        this.evaluation = eval;
        repaint();
    }

    public int getEvaluation() {
        return evaluation;
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        // Set the panel's background to a light blue.
        Color lightBlue = new Color(173, 216, 230);
        g.setColor(lightBlue);
        g.fillRect(0, 0, getWidth(), getHeight());

        // Draw border for the eval bar.
        int margin = 10;
        int barX = margin;
        int barY = margin;
        int barWidth = (getWidth() - 2 * margin) - 5;
        int barHeight = getHeight() - 2 * margin;

        g.setColor(Color.BLUE);
        g.drawRect(barX, barY, barWidth, barHeight);

        // Convert evaluation (-27 to 27) into a fraction (0.0 to 1.0).
        double fraction = (evaluation + 27) / 54.0;
        int whiteHeight = (int) (fraction * barHeight);

        // Fill from bottom: first draw white portion
        int whiteY = barY + barHeight - whiteHeight;
        g.setColor(Color.WHITE);
        g.fillRect(barX + 1, whiteY + 1, barWidth - 1, whiteHeight - 1);
        // Then fill the remaining top portion with black.
        g.setColor(Color.BLACK);
        g.fillRect(barX + 1, barY + 1, barWidth - 1, barHeight - whiteHeight - 1);

        // Draw the numeric evaluation value on the right side of the bar.
        g.setColor(Color.BLUE);
        g.setFont(new Font("SansSerif", Font.BOLD, 20));
        String evalStr = String.valueOf(evaluation);
        // Position it near the center-right of the panel.
        int textX = barX + barWidth + 3;
        int textY = barY + barHeight / 2 + 5;
        g.drawString(evalStr, textX, textY);
    }
}
