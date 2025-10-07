# Alpha-Beta TicTacToe 3D (5×5×5)

AI agents for **3D Tic-Tac-Toe** using **alpha-beta pruning** with forward pruning and heuristics.  
Includes a 100-game self-play tournament to compare strategies.

## File
- `tictactoe3d_alpha_beta.py` — heuristics, alpha-beta search, tournament runner.

## Dependency
Expects a `TicTacToe3D` environment (e.g., a course-provided `tdTTT` module).  
If not available, adapt the interface or provide a stub engine exposing:
- `board`, `players`, `current_player`, `winner`
- `get_legal_moves()`, `make_move(x,y,z)`, `check_current_turn()`

## Run
```bash
python tictactoe3d_alpha_beta.py
```
