## Introduction
Gomoku implemention of reinforcement learning by AlphaZero methods using pytorch c++ frontend.

## Usage
Enter `gomoku <command> -h` to see subcommand help in detail.
These are Gomoku subcommands:
```bash
   benchmark  benchmark between two mcts deep players
   help       shows help message and exits
   play       play with trained model
   train      train model from scatch or checkpoint
```

## Demo
Below shows a 8x8 game played between human (first hand, represented by `x`) and AI (represented by `o`). The model used has 64 filters, 3 residual blocks.
<br/>
![image](/src/renju/data/play_against_ai.gif)
