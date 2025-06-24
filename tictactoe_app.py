from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from kivy.uix.progressbar import ProgressBar
from kivy.uix.popup import Popup
import threading
from main import TicTacToe, QLearningAgent, MinimaxAgent, PrecomputedAgent


class GameButton(Button):

    def __init__(self, index, **kwargs):
        super().__init__(**kwargs)
        self.index = index
        self.font_size = 40
        self.background_color = (0.2, 0.6, 1, 1)


class TicTacToeApp(App):

    def __init__(self):
        super().__init__()
        self.game = TicTacToe()
        self.agent = QLearningAgent(alpha=0.3, gamma=0.95, epsilon=0.9)
        self.minimax_agent = MinimaxAgent()
        self.precomputed_agent = None  # Will be loaded when needed
        self.current_ai_type = "Q-Learning"  # Track which AI is active
        self.buttons = []
        self.current_player = 'X'  # Human is X, AI is O
        self.game_active = True
        self.training_popup = None

    def build(self):
        main_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

        # Title
        title = Label(text='Tic-Tac-Toe ML', font_size=24, size_hint_y=0.1)
        main_layout.add_widget(title)

        # Game status
        self.status_label = Label(text='Your turn (X)',
                                  font_size=18,
                                  size_hint_y=0.1)
        main_layout.add_widget(self.status_label)

        # Game board
        board_layout = GridLayout(cols=3, rows=3, spacing=5, size_hint_y=0.6)

        for i in range(9):
            btn = GameButton(i, text='')
            btn.bind(on_press=self.on_button_press)
            self.buttons.append(btn)
            board_layout.add_widget(btn)

        main_layout.add_widget(board_layout)

        # Control buttons
        control_layout = BoxLayout(orientation='horizontal',
                                   size_hint_y=0.2,
                                   spacing=10)

        reset_btn = Button(text='New Game',
                           background_color=(0.2, 0.8, 0.2, 1))
        reset_btn.bind(on_press=self.reset_game)
        control_layout.add_widget(reset_btn)

        train_btn = Button(text='Train AI',
                           background_color=(0.8, 0.2, 0.8, 1))
        train_btn.bind(on_press=self.train_ai)
        control_layout.add_widget(train_btn)

        # Automation controls
        auto_layout = BoxLayout(orientation='horizontal',
                                size_hint_y=0.15,
                                spacing=5)

        self.auto_play_btn = Button(text='Auto Play',
                                    background_color=(0.2, 0.8, 0.8, 1))
        self.auto_play_btn.bind(on_press=self.toggle_auto_play)
        auto_layout.add_widget(self.auto_play_btn)

        self.speed_btn = Button(text='Speed: 1x',
                                background_color=(0.8, 0.8, 0.2, 1))
        self.speed_btn.bind(on_press=self.cycle_speed)
        auto_layout.add_widget(self.speed_btn)

        # AI selection controls
        ai_layout = BoxLayout(orientation='horizontal',
                              size_hint_y=0.1,
                              spacing=5)

        self.ai_type_btn = Button(text='AI: Q-Learning',
                                  background_color=(0.6, 0.4, 0.8, 1))
        self.ai_type_btn.bind(on_press=self.cycle_ai_type)
        ai_layout.add_widget(self.ai_type_btn)

        precompute_btn = Button(text='Load Perfect AI',
                                background_color=(0.8, 0.6, 0.2, 1))
        precompute_btn.bind(on_press=self.load_perfect_ai)
        ai_layout.add_widget(precompute_btn)

        main_layout.add_widget(control_layout)
        main_layout.add_widget(auto_layout)
        main_layout.add_widget(ai_layout)

        # Auto play variables
        self.auto_playing = False
        self.play_speeds = [0.5, 1.0, 2.0, 3.0]
        self.current_speed_index = 1
        self.auto_event = None
        self.games_played = 0
        self.ai_wins = 0
        self.human_wins = 0
        self.draws = 0

        return main_layout

    def on_button_press(self, button):
        if not self.game_active or button.text != '':
            return

        # Human move
        self.make_move(button.index, 'X')

        if self.game_active and not self.game.is_game_over():
            # AI move
            Clock.schedule_once(self.ai_move, 0.5)

    def make_move(self, position, player):
        if position in self.game.available_actions():
            self.game.step(position, player)
            self.buttons[position].text = player
            self.buttons[position].color = (1, 0, 0,
                                            1) if player == 'X' else (0, 0, 1,
                                                                      1)

            if self.game.is_game_over():
                self.end_game()
            else:
                self.current_player = 'O' if player == 'X' else 'X'
                self.update_status()

    def ai_move(self, dt):
        if not self.game_active:
            return

        available = self.game.available_actions()
        if available:
            # Use more strategic decision making
            action = self.choose_best_move(available)
            self.make_move(action, 'O')

    def update_status(self):
        if self.current_player == 'X':
            self.status_label.text = 'Your turn (X)'
        else:
            self.status_label.text = 'AI thinking... (O)'

    def end_game(self):
        self.game_active = False
        winner_text = self.game.get_winner_text()
        if winner_text:
            self.status_label.text = winner_text
        else:
            self.status_label.text = 'Game Over'

    def reset_game(self, button):
        self.game.reset()
        self.current_player = 'X'
        self.game_active = True

        for btn in self.buttons:
            btn.text = ''
            btn.color = (1, 1, 1, 1)

        self.status_label.text = 'Your turn (X)'

    def train_ai(self, button):
        content = BoxLayout(orientation='vertical', spacing=10, padding=10)

        label = Label(
            text=
            'Training AI against random opponent...\nThis may take a moment.',
            text_size=(300, None),
            halign='center')
        content.add_widget(label)

        progress = ProgressBar(max=100)
        content.add_widget(progress)

        close_btn = Button(text='Training...', size_hint_y=0.3, disabled=True)
        content.add_widget(close_btn)

        self.training_popup = Popup(title='Training AI',
                                    content=content,
                                    size_hint=(0.8, 0.6),
                                    auto_dismiss=False)
        self.training_popup.open()

        # Start training in background thread
        threading.Thread(target=self.run_training,
                         args=(progress, close_btn, label)).start()

    def toggle_auto_play(self, button):
        self.auto_playing = not self.auto_playing

        if self.auto_playing:
            self.auto_play_btn.text = 'Stop Auto'
            self.auto_play_btn.background_color = (0.8, 0.2, 0.2, 1)
            self.start_auto_play()
        else:
            self.auto_play_btn.text = 'Auto Play'
            self.auto_play_btn.background_color = (0.2, 0.8, 0.8, 1)
            self.stop_auto_play()

    def cycle_speed(self, button):
        self.current_speed_index = (self.current_speed_index + 1) % len(
            self.play_speeds)
        speed = self.play_speeds[self.current_speed_index]
        self.speed_btn.text = f'Speed: {speed}x'

        if self.auto_playing:
            self.stop_auto_play()
            self.start_auto_play()

    def start_auto_play(self):
        speed = self.play_speeds[self.current_speed_index]
        interval = 1.0 / speed
        self.auto_event = Clock.schedule_interval(self.auto_play_step,
                                                  interval)
        self.reset_game(None)

    def stop_auto_play(self):
        if self.auto_event:
            self.auto_event.cancel()
            self.auto_event = None

    def auto_play_step(self, dt):
        if not self.game_active:
            # Game ended, start new one
            self.update_stats()
            Clock.schedule_once(lambda dt: self.reset_game(None), 0.3)
            return

        available = self.game.available_actions()
        if not available:
            return

        if self.current_player == 'X':
            # AI vs AI mode - use trained agent for X
            action = self.agent.choose_action(self.game.get_state(), available)
        else:
            # Random player for O
            action = self.choose_random_action(available)

        self.make_move(action, self.current_player)

    def choose_random_action(self, available_actions):
        import random
        return random.choice(available_actions)

    def update_stats(self):
        self.games_played += 1
        winner_text = self.game.get_winner_text()

        if "X Wins!" in winner_text:
            self.ai_wins += 1
        elif "O Wins!" in winner_text:
            self.human_wins += 1
        else:
            self.draws += 1

        win_rate = (self.ai_wins /
                    self.games_played) * 100 if self.games_played > 0 else 0
        self.status_label.text = f'Games: {self.games_played} | AI Wins: {self.ai_wins} | Rate: {win_rate:.1f}'

    def train_ai_simple(self):
        """Improved training with better reward shaping and exploration decay"""
        games_won = 0
        games_drawn = 0
        total_games = 2000  # More training games
        initial_epsilon = self.agent.epsilon

        for episode in range(total_games):
            # Decay exploration over time
            self.agent.epsilon = initial_epsilon * (0.995**episode)

            game = TicTacToe()
            agent_moves = []  # Store (state, action) for agent
            move_count = 0

            while not game.is_game_over():
                state = game.get_state()
                available = game.available_actions()

                if move_count % 2 == 0:  # Agent's turn (X)
                    action = self.agent.choose_action(state, available)
                    agent_moves.append((state, action))
                    game.step(action, 'X')
                else:  # Random opponent's turn (O)
                    import random
                    action = random.choice(available)
                    game.step(action, 'O')

                move_count += 1

            # Calculate rewards with better shaping
            final_state = game.get_state()
            if game.check_winner('X') > 0:
                games_won += 1
                final_reward = 10.0  # Big reward for winning
            elif game.check_winner('O') > 0:
                final_reward = -10.0  # Big penalty for losing
            elif game.is_draw():
                games_drawn += 1
                final_reward = 1.0  # Small reward for draw
            else:
                final_reward = 0.0  # Should not happen

            # Reward shaping: give intermediate rewards
            for i, (state, action) in enumerate(agent_moves):
                # Calculate intermediate rewards
                immediate_reward = self.calculate_move_reward(
                    state, action, final_state, final_reward)

                if i == len(agent_moves) - 1:
                    # Final move
                    self.agent.learn(state, action, immediate_reward,
                                     final_state, [])
                else:
                    # Get next state after opponent's move
                    if i + 1 < len(agent_moves):
                        next_state = agent_moves[i + 1][0]
                    else:
                        next_state = final_state

                    next_available = []
                    if not game.is_game_over():
                        temp_game = TicTacToe()
                        temp_game.board = list(next_state)
                        next_available = temp_game.available_actions()

                    self.agent.learn(state, action, immediate_reward,
                                     next_state, next_available)

        # Reset epsilon for gameplay
        self.agent.epsilon = 0.1
        return games_won / total_games

    def calculate_move_reward(self, state, action, final_state, final_reward):
        """Give intermediate rewards for good moves"""
        # Convert state to board for analysis
        board = list(state)

        # Check if this move creates a winning opportunity
        board[action] = 'X'
        temp_game = TicTacToe()
        temp_game.board = board

        # Reward for winning moves
        if temp_game.check_winner('X') > 0:
            return 5.0

        # Reward for blocking opponent wins
        board[action] = 'O'  # Test if opponent could win here
        if temp_game.check_winner('O') > 0:
            return 3.0

        # Small reward for center and corners
        if action == 4:  # Center
            return 0.5
        elif action in [0, 2, 6, 8]:  # Corners
            return 0.3

        # Base reward from final outcome
        return final_reward * 0.1

    def cycle_ai_type(self, button):
        """Cycle between different AI types"""
        ai_types = ["Q-Learning", "Minimax", "Precomputed"]
        current_index = ai_types.index(self.current_ai_type)
        self.current_ai_type = ai_types[(current_index + 1) % len(ai_types)]

        if self.current_ai_type == "Precomputed" and self.precomputed_agent is None:
            self.current_ai_type = "Q-Learning"  # Skip if not loaded

        self.ai_type_btn.text = f'AI: {self.current_ai_type}'

    def load_perfect_ai(self, button):
        """Load the precomputed perfect AI"""
        if self.precomputed_agent is None:
            button.text = "Loading..."
            button.disabled = True

            def load_in_background():
                self.precomputed_agent = PrecomputedAgent()
                Clock.schedule_once(lambda dt: self.finish_loading(button), 0)

            threading.Thread(target=load_in_background).start()

    def finish_loading(self, button):
        button.text = "Perfect AI Loaded!"
        button.disabled = False
        button.background_color = (0.2, 0.8, 0.2, 1)

    def choose_best_move(self, available_actions):
        """Choose move based on selected AI type"""
        if self.current_ai_type == "Minimax":
            return self.minimax_agent.choose_action(self.game, 'O')
        elif self.current_ai_type == "Precomputed" and self.precomputed_agent:
            return self.precomputed_agent.choose_action(self.game, 'O')
        else:  # Q-Learning with strategy
            current_state = self.game.get_state()

            # 1. Check for winning move
            for action in available_actions:
                temp_game = self.game.copy()
                temp_game.step(action, 'O')
                if temp_game.check_winner('O') > 0:
                    return action

            # 2. Check for blocking opponent's win
            for action in available_actions:
                temp_game = self.game.copy()
                temp_game.step(action, 'X')
                if temp_game.check_winner('X') > 0:
                    return action

            # 3. Use Q-learning for other moves
            return self.agent.choose_action(current_state, available_actions)

    def run_training(self, progress, close_btn, label):

        def update_progress(dt):
            progress.value = min(progress.value + 1, 100)

        # Schedule progress updates
        progress_event = Clock.schedule_interval(update_progress, 0.05)

        # Train the agent
        win_rate = self.train_ai_simple()

        # Update UI on main thread
        def finish_training(dt):
            progress_event.cancel()
            progress.value = 100
            label.text = f'Training Complete!\nWin rate: {win_rate:.1%}\nAgent learned {len(self.agent.q)} states'
            close_btn.text = 'Close'
            close_btn.disabled = False
            close_btn.bind(on_press=lambda x: self.training_popup.dismiss())

        Clock.schedule_once(finish_training, 0.1)


if __name__ == '__main__':
    TicTacToeApp().run()
