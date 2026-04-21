# Interactive Q-learning vs SARSA Web Lab

This project provides a fully interactive Web Application simulating **Q-learning** and **SARSA** reinforcement learning algorithms in the classic 4x12 **Cliff Walking** environment. 

Developed entirely using JavaScript, HTML5, and TailwindCSS, this tool visualizes grid-world behaviors matching the rigorous specifications of Sutton & Barto's *Reinforcement Learning: An Introduction*.

## Live Demo
Check out the simulation running on GitHub Pages:
**[View Interactive RL Lab](https://JodyYan.github.io/QlearningSarsaProject/)**

## Setup & Parameters
According to the textbook standard:
- **State Space**: 4x12 grid. Start at `(3,0)`. Goal at `(3,11)`. Cliff from `(3,1)` to `(3,10)`.
- **Epsilon (ε)**: `0.1` (10% exploration rate).
- **Alpha (α)**: `0.1` (Learning rate).
- **Gamma (γ)**: `0.9` (Discount factor).
- **Rewards**: 
    - `-1` for every safe transition.
    - `-100` for falling into the cliff (the agent gets reset to the start).

## Features
1. **Real-time Agent Simulation**: Watch the agent (blue dot) explore the grid visually.
2. **Dynamic Policy Layout**: The learned policy (arrows) updates continuously as the agent learns.
3. **Reward Charting**: Chart.js integration dynamically charts the `Episode` vs `Sum of Rewards` convergence.
4. **Speed Control**: Use the slider to slow down execution to study exploration vs exploitation, or max it out to fast-forward `500 episodes` rapidly!
5. **PDF Report Export**: Built with `jsPDF` and `html2canvas` to automatically capture the simulation results, charts, and theory explanations into a downloadable PDF summary!

## Theory Validated
- **Q-learning**: Off-policy configuration learns the optimal path (kissing the cliff edge). When evaluating at the end, the path length produces an optimal reward `-13`. However, the constant random exploration during learning causes the agent to fall off the cliff often, making the training reward average lower.
- **SARSA**: On-policy configuration learns an inherently safer but suboptimal path by walking upwards first, keeping its distance from the cliff. When simulated, this path is longer (reward `-17`), but because it factors in the risk of exploration, it results in fewer cliff falls and a smoother convergence curve during training.

## Usage
Simply clone the repository and open `index.html` in any modern web browser to interact. No server needed!
