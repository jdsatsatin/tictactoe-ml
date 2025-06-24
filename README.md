# Tic-Tac-Toe Machine Learning Agent

This project is a Tic-Tac-Toe game built with Python and the Kivy framework. It features a machine learning agent that learns to play the game through Q-learning. You can play against the agent or watch it play against a random opponent to see its learned strategy in action.

## How the AI Works

The agent's intelligence is a combination of rule-based logic and a learned strategy from a Q-learning model. This hybrid approach ensures the agent is both fundamentally sound and strategically smart.

### 1. Training Phase

- When you click **"Train AI"** (or when the app first starts), the agent plays tens of thousands of games against a random opponent.
- It plays as both 'X' and 'O' to learn both offensive and defensive strategies from both sides of the board.
- **Reward System:**
  - It receives a large **positive reward** for winning.
  - It receives a large **negative reward** (a penalty) for losing.
  - It receives a small **positive reward** for a draw, encouraging it to avoid losses.
- This process builds a "Q-table," which is essentially a memory of which moves are good or bad in any given board state.

### 2. Decision-Making Logic

When it's the agent's turn to make a move, it follows a clear, hierarchical strategy to ensure it plays as intelligently as possible:

1. **Rule 1: Check for an Immediate Win**
    - The agent first scans the board to see if it can place its piece ('O') in a square to win the game instantly.
    - If a winning move is found, it will **always** take it.

2. **Rule 2: Block the Opponent's Win**
    - If the agent cannot win on its current turn, it then checks if the opponent ('X') has a potential winning move on their *next* turn.
    - If it finds such a threat, it will **always** play its piece in that spot to block the opponent.

3. **Rule 3: Use the Learned ML Strategy**
    - If there are no immediate winning moves or necessary blocks, the agent consults its Q-table.
    - It looks at the current board state and chooses the available move that has the highest "Q-value" — the move it has learned will most likely lead to a win or a draw based on its extensive training.

This combination ensures the agent is never caught off-guard by obvious threats while still using its machine learning model for complex, strategic decisions.

## Application Features

- **New Game**: Resets the board to play a new game against the agent.
- **Train AI**: Initiates the training process to build or refine the agent's Q-table.
- **Auto Play**: Starts a continuous simulation where the ML Agent ('O') plays against a random player ('X'). This is useful for observing the agent's performance over many games.
- **Speed**: Adjusts the speed of the Auto Play simulation (1x, 2x, 4x, 8x).
- **Statistics Bar**: Shows real-time statistics for all completed games, including win/draw/loss rates for both the agent and the player.
