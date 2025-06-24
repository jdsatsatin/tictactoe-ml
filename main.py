from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from kivy.uix.progressbar import ProgressBar
from kivy.uix.popup import Popup
import threading
from tto import TicTacToe, QLearningAgent


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
        self.agent = QLearningAgent(alpha=0.3, gamma=0.95, epsilon=0.1)
        self.buttons = []
        self.current_player = 'X'  # Human/Random is X, Agent is O
        self.game_active = False
        self.training_popup = None
        self.auto_playing = False
        self.play_speeds = [1.0, 2.0, 4.0, 8.0]
        self.current_speed_index = 0
        self.auto_event = None
        self.games_played = 0
        self.agent_o_wins = 0  # Agent is always O
        self.draws = 0
        self.player_x_wins = 0  # Track Player X wins (Human or Random)
        self.stats_label = None  # Persistent stats label

    def build(self):
        main_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

        # Title
        title = Label(text='Tic-Tac-Toe ML', font_size=24, size_hint_y=0.1)
        main_layout.add_widget(title)

        # Game status
        self.status_label = Label(text='Training AI...',
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

        # Persistent statistics label
        self.stats_label = Label(text=self.get_stats_text(),
                                 font_size=16,
                                 size_hint_y=0.1)
        main_layout.add_widget(self.stats_label)

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
        train_btn.bind(on_press=self.train_ai_popup)
        control_layout.add_widget(train_btn)

        self.auto_play_btn = Button(text='Auto Play',
                                    background_color=(0.2, 0.8, 0.8, 1))
        self.auto_play_btn.bind(on_press=self.toggle_auto_play)
        control_layout.add_widget(self.auto_play_btn)

        self.speed_btn = Button(text='Speed: 1x',
                                background_color=(0.8, 0.8, 0.2, 1))
        self.speed_btn.bind(on_press=self.cycle_speed)
        control_layout.add_widget(self.speed_btn)

        main_layout.add_widget(control_layout)

        return main_layout

    def get_stats_text(self):
        win_rate = (self.agent_o_wins /
                    self.games_played) * 100 if self.games_played > 0 else 0
        draw_rate = (self.draws /
                     self.games_played) * 100 if self.games_played > 0 else 0
        x_win_rate = (self.player_x_wins /
                      self.games_played) * 100 if self.games_played > 0 else 0
        return (
            f"Games: {self.games_played} | Agent (O) Wins: {self.agent_o_wins} ({win_rate:.1f}%) | "
            f"Draws: {self.draws} ({draw_rate:.1f}%) | Player (X) Wins: {self.player_x_wins} ({x_win_rate:.1f}%)"
        )

    def update_stats_label(self):
        if self.stats_label:
            self.stats_label.text = self.get_stats_text()

    def on_start(self):
        self.show_training_popup()

    def show_training_popup(self):
        content = BoxLayout(orientation='vertical', spacing=10, padding=10)
        label = Label(text='Training AI...\nPlease wait.',
                      text_size=(300, None),
                      halign='center')
        content.add_widget(label)
        progress = ProgressBar(max=100)
        content.add_widget(progress)
        self.training_popup = Popup(title='Training AI',
                                    content=content,
                                    size_hint=(0.8, 0.4),
                                    auto_dismiss=False)
        self.training_popup.open()
        threading.Thread(target=self.run_training,
                         args=(progress, label)).start()

    def train_ai_popup(self, button):
        self.show_training_popup()

    def run_training(self, progress, label):
        total_games = 20000

        # Simulate progress as the training function is blocking
        def progress_updater(dt):
            if progress.value < 95:
                progress.value += 1

        progress_event = Clock.schedule_interval(progress_updater, 0.05)

        # Run the actual, improved training from main.py
        stats = self.agent.train_against_random(episodes=total_games)

        progress_event.cancel()
        win_rate = stats['win'] / total_games
        draw_rate = stats['draw'] / total_games

        def finish(dt):
            progress.value = 100
            label.text = (
                f'Training Complete!\n'
                f'Win Rate: {win_rate:.1%}, Draw Rate: {draw_rate:.1%}\n'
                f'Agent learned {len(self.agent.q)} states')
            # Delay dismissal to allow user to read the message
            Clock.schedule_once(lambda d: self.training_popup.dismiss(), 2.5)
            self.reset_game(None)
            self.game_active = True

        Clock.schedule_once(finish, 0.1)

    def on_button_press(self, button):
        if not self.game_active or button.text != '':
            return

        # Human move
        self.make_move(button.index, 'X')

        if self.game_active and not self.game.is_game_over():
            # AI move
            # Use the learned model for AI's move
            Clock.schedule_once(self.ai_move, 0.1)

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
        if not available:
            return

        # Rule 1: Check for a winning move for Agent 'O'
        for move in available:
            temp_game = self.game.copy()
            temp_game.step(move, 'O')
            if temp_game.check_winner('O'):
                self.make_move(move, 'O')
                return

        # Rule 2: Check to block opponent's ('X') winning move
        for move in available:
            temp_game = self.game.copy()
            temp_game.step(move, 'X')
            if temp_game.check_winner('X'):
                self.make_move(move, 'O')  # AI plays 'O' to block
                return

        # Rule 3: If no immediate win or block, use the Q-learning agent
        action = self.agent.choose_action(self.game.get_state(), available)
        if action is not None:
            self.make_move(action, 'O')

    def update_status(self):
        if self.auto_playing:
            if self.current_player == 'X':
                self.status_label.text = "Random Player's Turn (X)"
            else:
                self.status_label.text = "Agent's Turn (O)"
        else:
            if self.current_player == 'X':
                self.status_label.text = 'Your turn (X)'
            else:
                self.status_label.text = 'Agent thinking... (O)'

    def end_game(self):
        self.game_active = False
        winner_text = self.game.get_winner_text()
        self.status_label.text = winner_text
        # Update stats for all games (human, AI, draw)
        self.games_played += 1
        if "X Wins!" in winner_text:
            self.player_x_wins += 1
        elif "O Wins!" in winner_text:
            self.agent_o_wins += 1
        elif "Draw" in winner_text:
            self.draws += 1
        self.update_stats_label()

    def reset_game(self, button):
        self.game.reset()
        self.current_player = 'X'
        self.game_active = True
        for btn in self.buttons:
            btn.text = ''
            btn.color = (1, 1, 1, 1)
        self.update_status()
        # Do not reset stats here

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
        self.speed_btn.text = f'Speed: {int(speed)}x'

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
            Clock.schedule_once(lambda dt: self.reset_game(None), 0.05)
            return

        available = self.game.available_actions()
        if not available:
            return

        if self.current_player == 'X':  # Random Player
            import random
            action = random.choice(available)
        else:  # Agent 'O'
            action = self.agent.choose_action(self.game.get_state(), available)

        self.make_move(action, self.current_player)

    # Remove update_stats (now handled in end_game)


if __name__ == '__main__':
    TicTacToeApp().run()
