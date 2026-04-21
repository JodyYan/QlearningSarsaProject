# Q-learning vs SARSA on Cliff Walking

This project provides a comprehensive comparison between two fundamental Reinforcement Learning algorithms: **Q-learning** (off-policy) and **SARSA** (on-policy), applied to the classic Gridworld "Cliff Walking" environment.

## Overview
- **Environment**: 4x12 Cliff Walking Gridworld.
- **Parameters**: ε (Epsilon) = 0.1, α (Alpha) = 0.5, γ (Gamma) = 1.0. 
- **Episodes & Runs**: Average results are taken over 50 independent runs, with each run simulating 500 episodes.

### Deliverables
1. `main.py`: Contains the logic for the RL agents and environment. Runs the experiments and generates the resulting maps/graphs.
2. `reward_plot.png`: Smoothing plot of the sum of rewards per episode highlighting the statistical difference in convergence between the two algorithms.
3. `policy_map.png`: A visual grid map showing the final policies learned by both algorithms.
4. `report.pdf`: A technical report detailing the mathematical, analytical, and practical differences between Q-learning and SARSA.
5. `index.html`: A static web page designed to host the project on GitHub Pages, including embeded results and access to the technical report.

## How to Run Locally

### Requirements
You will need `Python 3` and the following libraries:
- `numpy`
- `matplotlib`
- `fpdf2`

Install them via pip:
```bash
pip install numpy matplotlib fpdf2
```

### Execution
Run the script to run the 50 trial runs (it may take a few seconds). 
```bash
python main.py
```
This will generate `reward_plot.png`, `policy_map.png`, and `report.pdf` in your current directory.

## GitHub Pages Deployment
If you fork/download this repository, it is ready to be hosted as a GitHub page. See the instructions below.
