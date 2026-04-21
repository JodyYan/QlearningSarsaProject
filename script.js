// script.js
const ROWS = 4;
const COLS = 12;
const START = {r: 3, c: 0};
const GOAL = {r: 3, c: 11};
const ACTIONS = [0, 1, 2, 3]; // 0: Up, 1: Right, 2: Down, 3: Left
const ARROW_SYMBOLS = ['↑', '→', '↓', '←'];

// Hyperparameters
const EPSILON = 0.1;
const ALPHA = 0.1;
const GAMMA = 0.9;
const MAX_EPISODES = 500;

// State Variables
let qTable = [];
let episodeCount = 0;
let currentRewards = [];
let isRunning = false;
let isPaused = false;
let loopContext = null;

// UI Elements
const gridWorld = document.getElementById('gridWorld');
const epCountEl = document.getElementById('epCount');
const curRewardEl = document.getElementById('curReward');
const algoSelect = document.getElementById('algoSelect');
const speedRange = document.getElementById('speedRange');
const startBtn = document.getElementById('startBtn');
const pauseBtn = document.getElementById('pauseBtn');
const resetBtn = document.getElementById('resetBtn');

// Chart instance
let rewardChart;

function initChart() {
    const ctx = document.getElementById('rewardChart').getContext('2d');
    if (rewardChart) rewardChart.destroy();
    
    rewardChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Sarsa Vs. Q-Learning Cliff Walking',
                data: [],
                borderColor: '#14b8a6', // default color, will update
                borderWidth: 2,
                fill: false,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { title: { display: true, text: 'Episodes' } },
                y: { title: { display: true, text: 'Sum of Rewards during episode' }, min: -100, max: 0 }
            },
            animation: false
        }
    });
}

function initGrid() {
    gridWorld.innerHTML = '';
    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
            let cell = document.createElement('div');
            cell.className = 'grid-cell';
            cell.id = `cell-${r}-${c}`;
            
            if (r === START.r && c === START.c) {
                cell.classList.add('cell-start');
                cell.innerText = 'Start';
            } else if (r === GOAL.r && c === GOAL.c) {
                cell.classList.add('cell-goal');
                cell.innerText = 'Goal';
            } else if (r === 3 && c > 0 && c < 11) {
                cell.classList.add('cell-cliff');
            } else {
                let arrow = document.createElement('span');
                arrow.className = 'arrow';
                arrow.id = `arrow-${r}-${c}`;
                cell.appendChild(arrow);
            }
            gridWorld.appendChild(cell);
        }
    }
    
    // Add agent
    let agent = document.createElement('div');
    agent.id = 'agent';
    gridWorld.appendChild(agent);
    moveAgentUI(START.r, START.c);
}

function moveAgentUI(r, c) {
    const agent = document.getElementById('agent');
    // Cell size is % based on grid rows/cols
    const cellWidth = 100 / COLS;
    const cellHeight = 100 / ROWS;
    
    // Center in cell
    agent.style.left = `calc(${c * cellWidth}% + ${(cellWidth / 2)}% - 30%)`; // 30% is half of 60% agent width relative to cell
    agent.style.top = `calc(${r * cellHeight}% + ${(cellHeight / 2)}% - 30%)`;
}

function updatePolicyUI() {
    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
            if ((r===START.r && c===START.c) || (r===GOAL.r && c===GOAL.c) || (r===3 && c>0 && c<11)) continue;
            let arrowEl = document.getElementById(`arrow-${r}-${c}`);
            if (arrowEl) {
                let maxQ = Math.max(...qTable[r][c]);
                if (maxQ === 0) {
                    arrowEl.innerText = ''; // Unexplored mostly
                } else {
                    let bestA = qTable[r][c].indexOf(maxQ);
                    arrowEl.innerText = ARROW_SYMBOLS[bestA];
                }
            }
        }
    }
}

function defaultQTable() {
    let q = [];
    for (let r = 0; r < ROWS; r++) {
        let row = [];
        for (let c = 0; c < COLS; c++) {
            row.push([0,0,0,0]);
        }
        q.push(row);
    }
    return q;
}

function resetEnv() {
    qTable = defaultQTable();
    episodeCount = 0;
    currentRewards = [];
    isRunning = false;
    isPaused = false;
    updateChartColor();
    initGrid();
    initChart();
    epCountEl.innerText = '0';
    curRewardEl.innerText = '0';
}

function stepEnv(state, action) {
    let r = state.r;
    let c = state.c;
    
    if (action === 0) r = Math.max(0, r - 1);
    else if (action === 1) c = Math.min(COLS - 1, c + 1);
    else if (action === 2) r = Math.min(ROWS - 1, r + 1);
    else if (action === 3) c = Math.max(0, c - 1);
    
    if (r === 3 && c > 0 && c < 11) {
        return { nextState: START, reward: -100, done: false };
    }
    if (r === GOAL.r && c === GOAL.c) {
        return { nextState: {r,c}, reward: -1, done: true }; 
    }
    
    return { nextState: {r,c}, reward: -1, done: false };
}

function chooseAction(state) {
    if (Math.random() < EPSILON) return Math.floor(Math.random() * 4);
    let maxQ = Math.max(...qTable[state.r][state.c]);
    let bestActions = [];
    for(let i=0; i<4; i++) if(qTable[state.r][state.c][i] === maxQ) bestActions.push(i);
    return bestActions[Math.floor(Math.random() * bestActions.length)];
}

async function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function updateChartColor() {
    if(!rewardChart) return;
    let isSarsa = algoSelect.value === 'sarsa';
    rewardChart.data.datasets[0].borderColor = isSarsa ? '#14b8a6' : '#ef4444'; // Cyan for Sarsa, Red for Q-learning
    rewardChart.update();
}

async function runEpisode() {
    let isSarsa = algoSelect.value === 'sarsa';
    let state = {r: START.r, c: START.c};
    let action = chooseAction(state);
    let done = false;
    let totalReward = 0;
    
    while (!done && isRunning) {
        if (isPaused) {
            await sleep(100);
            continue;
        }

        let res = stepEnv(state, action);
        let nextAction = chooseAction(res.nextState);
        
        let target;
        if (isSarsa) {
            target = res.reward + GAMMA * qTable[res.nextState.r][res.nextState.c][nextAction];
        } else {
            let maxQNext = Math.max(...qTable[res.nextState.r][res.nextState.c]);
            target = res.reward + GAMMA * maxQNext;
        }
        
        qTable[state.r][state.c][action] += ALPHA * (target - qTable[state.r][state.c][action]);
        totalReward += res.reward;
        
        state = res.nextState;
        if (isSarsa) action = nextAction;
        else action = chooseAction(state);

        let speed = parseInt(speedRange.value);
        if (speed < 100) {
            moveAgentUI(state.r, state.c);
            let delay = (100 - speed) * 2; 
            await sleep(delay);
        }
    }
    
    return totalReward;
}

async function startSimulation() {
    if (isRunning) {
        isPaused = false;
        return;
    }
    
    if (episodeCount >= MAX_EPISODES) resetEnv();
    
    isRunning = true;
    isPaused = false;
    updateChartColor();
    
    while (episodeCount < MAX_EPISODES && isRunning) {
        let totalReward = await runEpisode();
        
        if (!isRunning) break;

        episodeCount++;
        currentRewards.push(totalReward);
        
        epCountEl.innerText = episodeCount;
        curRewardEl.innerText = totalReward;
        
        rewardChart.data.labels.push(episodeCount);
        rewardChart.data.datasets[0].data.push(totalReward);
        
        // Optimize rendering by not updating every single frame if going fast
        if (parseInt(speedRange.value) < 100 || episodeCount % 10 === 0 || episodeCount === MAX_EPISODES) {
            updatePolicyUI();
            rewardChart.update();
        }
        
        // Hard-cap the visual limit for the y-axis dynamically based on reward curve
        let minR = Math.min(...currentRewards);
        if(minR < -200) rewardChart.options.scales.y.min = Math.max(minR, -500); 
    }
    
    isRunning = false;
}

startBtn.addEventListener('click', startSimulation);
pauseBtn.addEventListener('click', () => { isPaused = true; });
resetBtn.addEventListener('click', resetEnv);
algoSelect.addEventListener('change', resetEnv);

// jsPDF Export functionality
document.getElementById('downloadPdfBtn').addEventListener('click', async () => {
    const { jsPDF } = window.jspdf;
    const pdf = new jsPDF('p', 'pt', 'a4');
    
    pdf.setFontSize(22);
    pdf.text('Interactive RL Lab: Cliff Walking', 40, 50);
    
    pdf.setFontSize(14);
    pdf.text(`Algorithm Evaluated: ${algoSelect.value.toUpperCase()}`, 40, 80);
    pdf.text(`Episodes Reached: ${episodeCount} / 500`, 40, 100);
    pdf.text(`Final Episode Reward: ${currentRewards[currentRewards.length - 1] || 0}`, 40, 120);
    
    // Capture Grid and Chart
    const reportContent = document.getElementById('report-content');
    const canvas = await html2canvas(reportContent);
    const imgData = canvas.toDataURL('image/png');
    
    pdf.addImage(imgData, 'PNG', 40, 150, 515, (canvas.height * 515) / canvas.width);
    
    pdf.addPage();
    pdf.setFontSize(16);
    pdf.text('Theory & Discussion', 40, 50);
    pdf.setFontSize(12);
    let lines = pdf.splitTextToSize("Q-learning (Off-policy): Uses the max Q-value of the next state to update. It learns the absolutely shortest path right along the cliff edge. However, during training (when eps-greedy causes random actions), it occasionally falls off the cliff, resulting in high volatility and lower average reward during learning.\n\nSARSA (On-policy): Updates its Q-values using the actual action it takes next, factoring in the exploration policy. It quickly learns that positions near the cliff are dangerous due to the eps random chance. It converges to a safer, sub-optimal path farther away from the cliff edge, demonstrating a smoother and safer learning curve.", 515);
    pdf.text(lines, 40, 80);

    pdf.save('CliffWalking_Report.pdf');
});

// Initialize on load
resetEnv();
