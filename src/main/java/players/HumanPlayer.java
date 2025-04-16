package players;

import game.*;

public class HumanPlayer extends Player {
    public HumanPlayer(Board.Player symbol) {
        super(symbol);
    }

    @Override
    public void makeMove(Game game) {
    }

    @Override
    public String getName() {
        return "Human";
    }
}