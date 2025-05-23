#!/bin/bash
# Name of the tmux session
SESSION="training_session"

# Create a new detached tmux session
tmux new-session -d -s "$SESSION"

# Run the commands inside the tmux session
tmux send-keys -t "$SESSION" "zsh" C-m
tmux send-keys -t "$SESSION" "conda activate fiora" C-m
tmux send-keys -t "$SESSION" "cd /home/cis/heyo/ReadTheFudge/src" C-m
#tmux send-keys -t "$SESSION" "cd /home/cis/heyo/AV-Align/" C-m
#tmux send-keys -t "$SESSION" "cd /home/cis/heyo/Av-whatthefuck" C-m
tmux send-keys -t "$SESSION" "python train.py" C-m
