import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
import warnings

# Suppress warnings that might disrupt execution
warnings.filterwarnings('ignore')

# Environment Constants
ROWS = 4
COLS = 12
START = (3, 0)
GOAL = (3, 11)

# Actions: 0: Up, 1: Right, 2: Down, 3: Left
ACTIONS = [0, 1, 2, 3]

# Hyperparameters
EPSILON = 0.1
ALPHA = 0.5
GAMMA = 1.0
EPISODES = 500
RUNS = 50

def step(state, action):
    r, c = state
    if action == 0:   # Up
        r = max(0, r - 1)
    elif action == 1: # Right
        c = min(COLS - 1, c + 1)
    elif action == 2: # Down
        r = min(ROWS - 1, r + 1)
    elif action == 3: # Left
        c = max(0, c - 1)

    next_state = (r, c)

    # Check for cliff
    if r == 3 and 0 < c < 11:
        return START, -100, False
    # Check for goal
    if next_state == GOAL:
        return next_state, 0, True
        
    return next_state, -1, False

def choose_action(state, q_table, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(ACTIONS)
    values = q_table[state[0], state[1], :]
    return np.random.choice([a for a, v in enumerate(values) if v == np.max(values)])

def run_sarsa():
    q_table = np.zeros((ROWS, COLS, len(ACTIONS)))
    q_table[GOAL[0], GOAL[1], :] = 0  # Goal is terminal
    rewards = []

    for _ in range(EPISODES):
        state = START
        action = choose_action(state, q_table, EPSILON)
        total_reward = 0
        done = False

        while not done:
            next_state, reward, done = step(state, action)
            next_action = choose_action(next_state, q_table, EPSILON)
            
            td_target = reward + GAMMA * q_table[next_state[0], next_state[1], next_action] 
            td_error = td_target - q_table[state[0], state[1], action]
            q_table[state[0], state[1], action] += ALPHA * td_error

            total_reward += reward
            state = next_state
            action = next_action

        rewards.append(total_reward)
    return rewards, q_table

def run_q_learning():
    q_table = np.zeros((ROWS, COLS, len(ACTIONS)))
    q_table[GOAL[0], GOAL[1], :] = 0  # Goal is terminal
    rewards = []

    for _ in range(EPISODES):
        state = START
        total_reward = 0
        done = False

        while not done:
            action = choose_action(state, q_table, EPSILON)
            next_state, reward, done = step(state, action)
            
            best_next_action = np.argmax(q_table[next_state[0], next_state[1], :])
            td_target = reward + GAMMA * q_table[next_state[0], next_state[1], best_next_action] 

            td_error = td_target - q_table[state[0], state[1], action]
            q_table[state[0], state[1], action] += ALPHA * td_error

            total_reward += reward
            state = next_state

        rewards.append(total_reward)
    return rewards, q_table

def plot_rewards(sarsa_rewards, q_learning_rewards):
    plt.figure(figsize=(10, 6))
    
    # Smooth the curves a bit for better visualization like the prompt picture
    def moving_average(a, n=10) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    s_smooth = moving_average(sarsa_rewards, n=15)
    q_smooth = moving_average(q_learning_rewards, n=15)
    
    plt.plot(s_smooth, label='SARSA', color='c')
    plt.plot(q_smooth, label='Q-learning', color='r')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards during episode')
    plt.title('SARSA vs Q-Learning on Cliff Walking\n(Averaged over 50 runs, Smoothing=15)')
    plt.legend()
    plt.grid(True)
    plt.ylim(-100, 0)
    plt.savefig('reward_plot.png', bbox_inches='tight')
    plt.close()

def plot_policy(q_table, title, filename):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(0, COLS)
    ax.set_ylim(0, ROWS)
    ax.set_xticks(np.arange(0, COLS, 1))
    ax.set_yticks(np.arange(0, ROWS, 1))
    ax.grid(which='both', color='black', linestyle='-', linewidth=2)
    
    # Direction mappings
    dir_map = {0: (0, 0.4), 1: (0.4, 0), 2: (0, -0.4), 3: (-0.4, 0)}
    
    for r in range(ROWS):
        for c in range(COLS):
            # Coordinates for matplotlib plotting
            x = c + 0.5
            y = ROWS - 1 - r + 0.5
            
            if (r, c) == START:
                ax.text(x, y, 'Start', ha='center', va='center', fontsize=12)
                continue
            if (r, c) == GOAL:
                ax.text(x, y, 'Goal', ha='center', va='center', fontsize=12)
                continue
            if r == 3 and 0 < c < 11:
                ax.text(x, y, 'Cliff', ha='center', va='center', fontsize=12)
                ax.add_patch(plt.Rectangle((c, ROWS-1-r), 1, 1, facecolor='grey', alpha=0.5))
                continue
                
            best_action = np.argmax(q_table[r, c, :])
            dx, dy = dir_map[best_action]
            ax.arrow(x - dx*0.5, y - dy*0.5, dx, dy, head_width=0.2, head_length=0.2, fc='k', ec='k')

    ax.set_title(title)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def generate_pdf_report():
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Q-Learning vs SARSA: Cliff Walking Analysis', 0, 1, 'C')
    pdf.ln(10)
    
    # Theory Section: Off-policy vs On-policy
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '1. Off-policy vs. On-policy', 0, 1)
    pdf.set_font('Arial', '', 11)
    text1 = ("Q-learning is an off-policy algorithm. Its update rule uses the maximum action-value "
             "from the next state, effectively learning the value of the optimal policy independently of "
             "the agent's exploratory actions. SARSA is an on-policy algorithm. It updates its action-value "
             "based on the action actually taken during exploration, which means its learned policy reflects "
             "the exploratory policy being executed by the agent.")
    pdf.multi_cell(0, 6, text1)
    pdf.ln(5)
    
    # Risk Analysis
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '2. Risk Analysis', 0, 1)
    pdf.set_font('Arial', '', 11)
    text2 = ("During training, Q-learning's reward sum fluctuates more significantly and converges to "
             "a lower value compared to SARSA. This is because Q-learning learns the optimal (edge) path, "
             "but the eps-greedy exploration (eps=0.1) causes it to occasionally step off the cliff and incur "
             "the massive -100 penalty. SARSA, on the other hand, factors this exploration risk into its "
             "returns. It quickly learns that positions near the cliff are 'dangerous' because of the random "
             "exploration chance, so its learned policy moves safely away from the edge.")
    pdf.multi_cell(0, 6, text2)
    pdf.ln(5)

    # Optimal vs Sub-optimal but Safe
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '3. Optimal vs. Safe in Reality', 0, 1)
    pdf.set_font('Arial', '', 11)
    text3 = ("In simulated environments with test phases, Q-learning provides the absolute optimal "
             "trajectory if exploration is disabled after training. However, in real-world applications "
             "(e.g., autonomous driving or medical treatment), the environment itself may contain stochasticity, "
             "or testing may not be perfectly pristine. In these cases, a 'sub-optimal but safe' route as discovered "
             "by SARSA is vastly preferable. A small loss in efficiency is much better than risking catastrophic failure.")
    pdf.multi_cell(0, 6, text3)
    pdf.ln(10)
    
    # Add Plots
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Experimental Results', 0, 1, 'C')
    pdf.ln(5)
    
    # Assuming images are already generated and present in the directory
    pdf.image('reward_plot.png', x=15, w=180)
    pdf.ln(5)
    pdf.image('policy_map.png', x=15, w=180)

    pdf.output('report.pdf')

if __name__ == '__main__':
    all_sarsa_rewards = np.zeros((RUNS, EPISODES))
    all_q_rewards = np.zeros((RUNS, EPISODES))
    
    print("Running experiments...")
    # Because final policy is needed, we'll store the final Q-tables of the last run.
    # In RL, often the Q-table is averaged across runs or we just keep the last one.
    final_q_sarsa = None
    final_q_table = None

    for r in range(RUNS):
        print(f"Run {r+1}/{RUNS}")
        sarsa_r, q_sarsa = run_sarsa()
        q_r, q_q = run_q_learning()
        all_sarsa_rewards[r] = sarsa_r
        all_q_rewards[r] = q_r
        
        # Save last run's Q-table for plotting
        final_q_sarsa = q_sarsa
        final_q_table = q_q

    avg_sarsa = np.mean(all_sarsa_rewards, axis=0)
    avg_q = np.mean(all_q_rewards, axis=0)

    print("Plotting rewards...")
    plot_rewards(avg_sarsa, avg_q)
    
    # Policy Map should contain both on a single figure, or we can just plot them as a single combined image
    # Let's combine them into one image
    print("Plotting policies...")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Direction mappings
    dir_map = {0: (0, 0.4), 1: (0.4, 0), 2: (0, -0.4), 3: (-0.4, 0)}
    
    for i, (q_t, title) in enumerate([(final_q_table, 'Q-learning Policy'), (final_q_sarsa, 'SARSA Policy')]):
        ax = axes[i]
        ax.set_xlim(0, COLS)
        ax.set_ylim(0, ROWS)
        ax.set_xticks(np.arange(0, COLS, 1))
        ax.set_yticks(np.arange(0, ROWS, 1))
        ax.grid(which='both', color='black', linestyle='-', linewidth=2)
        
        for rt in range(ROWS):
            for c in range(COLS):
                x = c + 0.5
                y = ROWS - 1 - rt + 0.5
                
                if (rt, c) == START:
                    ax.text(x, y, 'Start', ha='center', va='center', fontsize=10)
                    continue
                if (rt, c) == GOAL:
                    ax.text(x, y, 'Goal', ha='center', va='center', fontsize=10)
                    continue
                if rt == 3 and 0 < c < 11:
                    ax.text(x, y, 'Cliff', ha='center', va='center', fontsize=10)
                    ax.add_patch(plt.Rectangle((c, ROWS-1-rt), 1, 1, facecolor='lightblue', alpha=0.5))
                    continue
                    
                best_action = np.argmax(q_t[rt, c, :])
                dx, dy = dir_map[best_action]
                ax.arrow(x - dx*0.5, y - dy*0.5, dx, dy, head_width=0.2, head_length=0.2, fc='k', ec='k')

        ax.set_title(title, fontsize=14, pad=10)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    plt.tight_layout()
    plt.savefig('policy_map.png', bbox_inches='tight')
    plt.close()

    print("Generating PDF report...")
    generate_pdf_report()
    
    print("Done! Check files.")
