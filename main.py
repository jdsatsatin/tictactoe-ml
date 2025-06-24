import random
import numpy as np


class TicTacToe:

    def __init__(self):
        self.board = [' '] * 9

    def reset(self):
        self.board = [' '] * 9
        return self.get_state()

    def get_state(self):
        return ''.join(self.board)

    def available_actions(self):
        return [i for i, x in enumerate(self.board) if x == ' ']

    def step(self, action, player):
        if self.board[action] == ' ':
            self.board[action] = player
            return self.get_state(), self.check_winner(player), False
        return self.get_state(), 0, True  # Invalid move

    def check_winner(self, player):
        combos = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7),
                  (2, 5, 8), (0, 4, 8), (2, 4, 6)]
        for c in combos:
            if self.board[c[0]] == self.board[c[1]] == self.board[
                    c[2]] == player:
                return 1
        return 0  # No win found - remove the draw logic from here

    def is_game_over(self):
        return self.check_winner('X') > 0 or self.check_winner(
            'O') > 0 or ' ' not in self.board

    def get_winner_text(self):
        if self.check_winner('X') > 0:
            return "X Wins!"
        elif self.check_winner('O') > 0:
            return "O Wins!"
        elif ' ' not in self.board:
            return "Draw!"
        return ""

    def is_draw(self):
        """Check if the game is a draw"""
        return ' ' not in self.board and self.check_winner(
            'X') == 0 and self.check_winner('O') == 0

    def get_empty_positions(self):
        return [i for i, x in enumerate(self.board) if x == ' ']

    def make_move(self, position, player):
        """Make a move and return if it was valid"""
        if self.board[position] == ' ':
            self.board[position] = player
            return True
        return False

    def undo_move(self, position):
        """Undo a move"""
        self.board[position] = ' '

    def copy(self):
        """Create a copy of the game"""
        new_game = TicTacToe()
        new_game.board = self.board.copy()
        return new_game


class QLearningAgent:

    def __init__(self, alpha=0.5, gamma=0.9, epsilon=0.1):
        self.q = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_q(self, state, action):
        return self.q.get((state, action), 0.0)

    def choose_action(self, state, actions):
        if random.random() < self.epsilon:
            return random.choice(actions)
        qs = [self.get_q(state, a) for a in actions]
        max_q = max(qs)
        return random.choice([a for a, q in zip(actions, qs) if q == max_q])

    def learn(self, state, action, reward, next_state, next_actions):
        old_q = self.get_q(state, action)
        future_q = max([self.get_q(next_state, a) for a in next_actions],
                       default=0)
        self.q[(
            state, action
        )] = old_q + self.alpha * (reward + self.gamma * future_q - old_q)

    def train_against_random(self, episodes=1000):
        game = TicTacToe()
        wins = 0

        for _ in range(episodes):
            game.reset()
            states = []
            actions = []

            while not game.is_game_over():
                state = game.get_state()
                available = game.available_actions()

                if len(states) % 2 == 0:  # Agent's turn (X)
                    action = self.choose_action(state, available)
                    states.append(state)
                    actions.append(action)
                else:  # Random opponent (O)
                    action = random.choice(available)

                game.step(action, 'X' if len(states) % 2 == 0 else 'O')

            # Learn from the game
            reward = game.check_winner('X')
            if reward > 0:
                wins += 1

            for i in range(len(states)):
                if i == len(states) - 1:
                    self.learn(states[i], actions[i], reward, game.get_state(),
                               [])
                else:
                    self.learn(states[i], actions[i], 0, states[i + 1],
                               game.available_actions())

        return wins / episodes


class MinimaxAgent:
    """Perfect play agent using minimax algorithm"""

    def __init__(self):
        self.memo = {}  # Memoization for faster computation

    def choose_action(self, game, player):
        """Choose the best move using minimax"""
        best_score = float('-inf')
        best_move = None

        for move in game.available_actions():
            game.make_move(move, player)
            score = self.minimax(game, 0, False, player)
            game.undo_move(move)

            if score > best_score:
                best_score = score
                best_move = move

        return best_move if best_move is not None else random.choice(
            game.available_actions())

    def minimax(self, game, depth, is_maximizing, player):
        """Minimax algorithm with memoization"""
        state = game.get_state()

        # Check memoization
        if (state, is_maximizing) in self.memo:
            return self.memo[(state, is_maximizing)]

        # Base cases
        if game.check_winner(player) > 0:
            result = 10 - depth  # Prefer faster wins
        elif game.check_winner('X' if player == 'O' else 'O') > 0:
            result = depth - 10  # Prefer slower losses
        elif game.is_game_over():
            result = 0  # Draw
        else:
            # Recursive case
            if is_maximizing:
                max_eval = float('-inf')
                for move in game.available_actions():
                    game.make_move(move, player)
                    eval_score = self.minimax(game, depth + 1, False, player)
                    game.undo_move(move)
                    max_eval = max(max_eval, eval_score)
                result = max_eval
            else:
                min_eval = float('inf')
                opponent = 'X' if player == 'O' else 'O'
                for move in game.available_actions():
                    game.make_move(move, opponent)
                    eval_score = self.minimax(game, depth + 1, True, player)
                    game.undo_move(move)
                    min_eval = min(min_eval, eval_score)
                result = min_eval

        # Memoize result
        self.memo[(state, is_maximizing)] = result
        return result


class PrecomputedAgent:
    """Agent that precomputes all possible game states"""

    def __init__(self):
        self.game_tree = {}
        self.build_complete_tree()

    def build_complete_tree(self):
        """Build complete game tree for perfect play"""
        print("Building complete game tree...")
        self._build_tree('         ', 'X')  # Empty board, X starts
        print(f"Built tree with {len(self.game_tree)} states")

    def _build_tree(self, state, player):
        """Recursively build the game tree"""
        if state in self.game_tree:
            return self.game_tree[state]

        game = TicTacToe()
        game.board = list(state)

        if game.is_game_over():
            if game.check_winner('X') > 0:
                score = 1
            elif game.check_winner('O') > 0:
                score = -1
            else:
                score = 0
            self.game_tree[state] = score
            return score

        available = game.available_actions()
        if player == 'X':  # Maximizing player
            best_score = float('-inf')
            for move in available:
                new_state = list(state)
                new_state[move] = 'X'
                new_state_str = ''.join(new_state)
                score = self._build_tree(new_state_str, 'O')
                best_score = max(best_score, score)
        else:  # Minimizing player
            best_score = float('inf')
            for move in available:
                new_state = list(state)
                new_state[move] = 'O'
                new_state_str = ''.join(new_state)
                score = self._build_tree(new_state_str, 'X')
                best_score = min(best_score, score)

        self.game_tree[state] = best_score
        return best_score

    def choose_action(self, game, player):
        """Choose best action from precomputed tree"""
        current_state = game.get_state()
        best_move = None
        best_score = float('-inf') if player == 'X' else float('inf')

        for move in game.available_actions():
            new_board = list(current_state)
            new_board[move] = player
            new_state = ''.join(new_board)

            if new_state in self.game_tree:
                score = self.game_tree[new_state]
                if player == 'X' and score > best_score:
                    best_score = score
                    best_move = move
                elif player == 'O' and score < best_score:
                    best_score = score
                    best_move = move

        return best_move if best_move is not None else random.choice(
            game.available_actions())
