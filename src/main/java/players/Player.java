package players;

import game.*;

public abstract class Player {
    protected Board.Player symbol;

    public Player(Board.Player symbol) {
        this.symbol = symbol;
    }

    public Board.Player getSymbol() {
        return symbol;
    }

    public abstract void makeMove(Game game);

    public abstract String getName();
}