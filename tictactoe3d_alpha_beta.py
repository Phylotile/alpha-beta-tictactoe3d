# -*- coding: utf-8 -*-
from tdTTT import TicTacToe3D
import copy         # For deep-copying game states.
import math         # Mathematical functions (if needed).
import random       # To add randomness in heuristic evaluations.

def generate_winning_lines(size=5):
    """
    Generate all winning lines for a 5x5x5 board.
    A winning line is any contiguous sequence of 5 cells in a straight line.
    """
    lines = []         # List to store all winning lines.
    directions = []    # Valid movement directions for a line.

    # Iterate over possible direction vectors, excluding the zero vector.
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                # To avoid duplicate lines, add only directions lexicographically > (0,0,0).
                if (dx, dy, dz) > (0, 0, 0):
                    directions.append((dx, dy, dz))

    # For each cell, try to construct a line in each valid direction.
    for x in range(size):
        for y in range(size):
            for z in range(size):
                for dx, dy, dz in directions:
                    line = []  # A candidate winning line.
                    for step in range(5):  # Winning line must have exactly 5 cells.
                        nx = x + step * dx
                        ny = y + step * dy
                        nz = z + step * dz
                        # Check bounds.
                        if 0 <= nx < size and 0 <= ny < size and 0 <= nz < size:
                            line.append((nx, ny, nz))
                        else:
                            break
                    if len(line) == 5:
                        lines.append(line)
    return lines

# Precompute winning lines once.
winning_lines = generate_winning_lines(size=5)

def clone_game(game):
    """
    Create a deep copy of the game state.
    Used for simulating moves without modifying the actual game.
    """
    new_game = TicTacToe3D(game.size)
    new_game.board = copy.deepcopy(game.board)
    new_game.current_player = game.current_player
    new_game.winner = game.winner
    return new_game

"""### Player 1: Alpha-beta forward pruning with heuristic 1
Explain here what your heuristic 1 is:
"""

# Heuristic 1 (Simple "Open Lines" Count):
#   This heuristic scans through every winning line. For each line, if the opponent is not present,
#   it adds the number of cells occupied by the player. Similarly, if the player is absent in the line,
#   it subtracts the count for the opponent. This provides a measure of the potential for winning.
def heuristic1(game, player):
    # Terminal state check.
    if game.winner is not None:
        if game.winner == player:
            return float('inf')
        elif game.winner == "Draw":
            return 0
        else:
            return float('-inf')
    opponent = 'O' if player == 'X' else 'X'
    score = 0
    # Evaluate each winning line.
    for line in winning_lines:
        # Get the contents of the line.
        line_values = [game.board[z][y][x] for (x, y, z) in line]
        # If the opponent is not present, count player's pieces.
        if opponent not in line_values:
            score += line_values.count(player)
        # If the player is not present, subtract opponent's count.
        if player not in line_values:
            score -= line_values.count(opponent)
    return score

"""### Player 2: Alpha-beta forward pruning with heuristic 2
Explain here what your heuristic 2 is:
"""

# Heuristic 2 (Piece Difference):
#   This heuristic is very simple: it counts the total number of pieces on the board for the player,
#   subtracting the number of opponent pieces. This gives a crude estimate of board control.
def heuristic2(game, player):
    # Terminal state check.
    if game.winner is not None:
        if game.winner == player:
            return float('inf')
        elif game.winner == "Draw":
            return 0
        else:
            return float('-inf')
    opponent = 'O' if player == 'X' else 'X'
    # Count all pieces for player and opponent.
    player_count = sum(cell == player for layer in game.board for row in layer for cell in row)
    opponent_count = sum(cell == opponent for layer in game.board for row in layer for cell in row)
    return player_count - opponent_count

def alpha_beta_search(game, depth, alpha, beta, maximizing, heuristic_eval, player, prune_moves=10):
    """
    Alpha-Beta search with forward pruning.
    Evaluates moves using a one-move lookahead to order them, then prunes the list.

    Parameters:
      - game: current game state.
      - depth: search depth.
      - alpha, beta: parameters for pruning.
      - maximizing: True if current node is maximizing.
      - heuristic_eval: function to evaluate game states.
      - player: the player's symbol for which we're evaluating.
      - prune_moves: number of moves to consider after ordering.

    Returns:
      (value, best_move)
    """
    # Check if the game has a winner (terminal state).
    if game.winner is not None:
        return heuristic_eval(game, player), None

    # Check if maximum depth is reached (base case for recursion).
    if depth == 0:
        return heuristic_eval(game, player), None

    # Get all legal moves available in the current game state.
    legal_moves = game.get_legal_moves()

    # If there are no legal moves, return the heuristic evaluation of the state.
    if not legal_moves:
        return heuristic_eval(game, player), None

    moves_scores = []  # List to store moves and their corresponding scores.

    # Evaluate each move using a one-move lookahead heuristic.
    for move in legal_moves:
        game_clone = clone_game(game)  # Create a copy of the current game state.

        try:
            game_clone.make_move(*move)  # Apply the move to the cloned game.
        except Exception:
            continue  # Skip this move if an error occurs.

        score = heuristic_eval(game_clone, player)  # Evaluate the resulting state.
        moves_scores.append((score, move))  # Store the move and its score.

    # Sort moves based on score to prioritize best moves first.
    if maximizing:
        moves_scores.sort(key=lambda x: x[0], reverse=True)  # Higher scores first.
    else:
        moves_scores.sort(key=lambda x: x[0])  # Lower scores first.

    # Forward pruning: Consider only the top 'prune_moves' best moves.
    ordered_moves = [move for (_, move) in moves_scores[:prune_moves]]

    best_move = None  # Variable to track the best move found.

    if maximizing:
        value = float('-inf')  # Initialize value to negative infinity (worst case for maximizer).

        for move in ordered_moves:
            child = clone_game(game)  # Clone game state for the move.

            try:
                child.make_move(*move)  # Apply the move.
            except Exception:
                continue  # Skip invalid moves.

            # Recursive call for minimizing player (opponent's turn).
            score, _ = alpha_beta_search(child, depth-1, alpha, beta, False, heuristic_eval, player, prune_moves)

            if score > value:  # Update best value and move if a better score is found.
                value = score
                best_move = move

            alpha = max(alpha, value)  # Update alpha (best max value found so far).

            if alpha >= beta:
                break  # Beta cutoff (pruning unnecessary branches).

        return value, best_move  # Return best score and move for maximizer.

    else:
        value = float('inf')  # Initialize value to positive infinity (worst case for minimizer).

        for move in ordered_moves:
            child = clone_game(game)  # Clone game state for the move.

            try:
                child.make_move(*move)  # Apply the move.
            except Exception:
                continue  # Skip invalid moves.

            # Recursive call for maximizing player (opponent's turn).
            score, _ = alpha_beta_search(child, depth-1, alpha, beta, True, heuristic_eval, player, prune_moves)

            if score < value:  # Update best value and move if a lower score is found.
                value = score
                best_move = move

            beta = min(beta, value)  # Update beta (best min value found so far).

            if beta <= alpha:
                break  # Alpha cutoff (pruning unnecessary branches).

        return value, best_move  # Return best score and move for minimizer.

"""AI for Player 1 (Exponential Heuristic)"""

class AI1:
    """
    AI using Heuristic 1 (Simple Open Lines Count).
    Uses alpha-beta search with forward pruning.
    """
    def __init__(self, depth=2, prune_moves=10):
        self.depth = depth              # Search depth.
        self.prune_moves = prune_moves  # How many moves to consider after ordering.
        self.heuristic = heuristic1       # Evaluation function.
        self.player = None              # Will be set to 'X' or 'O' at game start.

    def get_move(self, game):
        if self.player is None:
            self.player = game.players[game.current_player]
        _, move = alpha_beta_search(game, self.depth, float('-inf'), float('inf'), True, self.heuristic, self.player, self.prune_moves)
        return move

"""AI for Player 2 (Linear Heuristic)

"""

class AI2:
    """
    AI using Heuristic 2 (Piece Difference).
    Uses alpha-beta search with forward pruning.
    """
    def __init__(self, depth=2, prune_moves=10):
        self.depth = depth              # Search depth.
        self.prune_moves = prune_moves  # How many moves to consider after ordering.
        self.heuristic = heuristic2       # Evaluation function.
        self.player = None              # Will be set to 'X' or 'O' at game start.

    def get_move(self, game):
        if self.player is None:
            self.player = game.players[game.current_player]
        _, move = alpha_beta_search(game, self.depth, float('-inf'), float('inf'), True, self.heuristic, self.player, self.prune_moves)
        return move

"""### Competition

Run your competition here. Have the two players face each other 100 times, switching who starts each time. Print out the statistics in an easy-to-read format.
"""

def simulate_games(num_games=100):
    """
    Simulate a series of games between the two AIs.
    Alternates starting players to balance any first-move advantage.
    Returns a dictionary with win counts for each AI and draws.
    """
    results = {"Exponential": 0, "Linear": 0, "Draw": 0}

    for game_num in range(num_games):
        game = TicTacToe3D(size=5)  # Create a new 5x5x5 board.

        # Alternate starting players while keeping fixed heuristic labels.
        if game_num % 2 == 0:
            # Even games: Player 1 (Exponential) is 'X', Player 2 (Linear) is 'O'
            ai_expo = AI1(depth=1, prune_moves=10)   # Depth set to 1 for faster decision-making.
            ai_linear = AI2(depth=1, prune_moves=10)
            ai_expo.player = game.players[0]  # 'X'
            ai_linear.player = game.players[1]  # 'O'
        else:
            # Odd games: roles are swapped, but heuristic labels remain.
            ai_expo = AI1(depth=1, prune_moves=10)
            ai_linear = AI2(depth=1, prune_moves=10)
            ai_linear.player = game.players[0]  # 'X'
            ai_expo.player = game.players[1]    # 'O'

        # Play the game until a winner is declared or no moves remain.
        while game.winner is None:
            current_symbol = game.check_current_turn()
            if current_symbol == ai_expo.player:
                move = ai_expo.get_move(game)
            else:
                move = ai_linear.get_move(game)
            if move is None:
                break  # No legal moves.
            try:
                game.make_move(*move)
            except Exception:
                break

        # Record results using fixed heuristic labels.
        if game.winner == ai_expo.player:
            results["Exponential"] += 1
        elif game.winner == ai_linear.player:
            results["Linear"] += 1
        else:
            results["Draw"] += 1

        print(f"Game {game_num+1}: Winner: {game.winner}")
    return results


# Run Competition and Output Summary
try:
    results = simulate_games(num_games=100)  # Run 100 games.
except KeyboardInterrupt:
    print("Simulation interrupted by user.")

print("\nSimulation Results over 100 games:")
print(results)

"""# Output

Simulation Results over 100 games:
{'Exponential': 96, 'Linear': 4, 'Draw': 0}

# Analysis

This experiment involved two AI players competing in 3D Tic Tac Toe on a 5x5x5 board.
The win condition was to place 5 pieces in a straight line.

Player 1 (Exponential Heuristic):
    - Uses alpha-beta search with forward pruning and a simple open-lines counting heuristic.
    - Evaluates each potential winning line by counting the player's pieces if the opponent is absent.
    - Depth was set to 1 to improve computation speed.
    
Player 2 (Linear Heuristic):
    - Uses alpha-beta search with forward pruning and a piece difference heuristic.
    - Counts the difference between the player's and opponent's pieces on the board.
    - Depth was set to 1 for faster decision-making.

Depth Consideration:
    - A shallower search (depth=1) evaluates only immediate moves, which speeds up computation
      but may reduce long-term planning. A deeper search could yield more draws or balanced outcomes.

Competition:
    - The AIs played 100 games with alternating starting positions to balance any first-move advantage.
    - Summary statistics are printed above to show win distributions.
"""