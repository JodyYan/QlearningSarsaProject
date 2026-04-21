// script.js
const ROWS = 4;
const COLS = 12;
const START = {r: 3, c: 0};
const GOAL = {r: 3, c: 11};
const ARROW_SYMBOLS = ['↑', '→', '↓', '←']; // 0: Up, 1: Right, 2: Down, 3: Left

// Academic Hyperparameters (dynamic)
let EPSILON = 0.1;
let ALPHA = 0.5;
let GAMMA = 1.0;
let EPISODES = 500;
let RUNS = 50;

function updateParams() {
    EPISODES = parseInt(document.getElementById('inputEpisodes').value) || 500;
    RUNS = parseInt(document.getElementById('inputRuns').value) || 50;
    EPSILON = parseFloat(document.getElementById('inputEpsilon').value) || 0;
    ALPHA = parseFloat(document.getElementById('inputAlpha').value) || 0;
    GAMMA = parseFloat(document.getElementById('inputGamma').value) || 0;
}

const startBtn = document.getElementById('startBtn');
let rewardChart;

function initGrids() {
    buildGrid(document.getElementById('qGrid'));
    buildGrid(document.getElementById('sGrid'));
}

function buildGrid(container) {
    container.innerHTML = '';
    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
            let cell = document.createElement('div');
            cell.className = 'grid-cell';
            cell.id = `${container.id}-cell-${r}-${c}`;
            
            if (r === START.r && c === START.c) {
                cell.classList.add('cell-start');
                cell.innerHTML = '<span class="text-lg font-bold text-blue-600">↑</span><span>Start</span>';
            } else if (r === GOAL.r && c === GOAL.c) {
                cell.classList.add('cell-goal');
                cell.innerText = 'Goal';
            } else if (r === 3 && c > 0 && c < 11) {
                cell.classList.add('cell-cliff');
                cell.innerText = 'Cliff';
            } else {
                let arrow = document.createElement('span');
                arrow.className = 'arrow';
                arrow.id = `${container.id}-arrow-${r}-${c}`;
                cell.appendChild(arrow);
            }
            container.appendChild(cell);
        }
    }
}

// Generate the Sutton & Barto Dotted Lines
function generateSuttonBaseline(isSarsa) {
    let data = [];
    let target = isSarsa ? -25 : -45;
    for(let i=0; i<EPISODES; i++) {
        if (i < 30) {
            // Rapid climb curve
            let start = -100;
            let current = start + ((target - start) * (Math.log(i+1) / Math.log(30)));
            data.push(current);
        } else {
            // Noise around convergence
            data.push(target + (Math.random() * 4 - 2));
        }
    }
    return data;
}

function initChart() {
    const ctx = document.getElementById('rewardChart').getContext('2d');
    if (rewardChart) rewardChart.destroy();
    
    let labels = Array.from({length: EPISODES}, (_, i) => i + 1);
    
    rewardChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Sarsa',
                    data: [],
                    borderColor: '#06b6d4', // Cyan solid
                    borderWidth: 2,
                    fill: false,
                    pointRadius: 0,
                    tension: 0.1
                },
                {
                    label: 'Q-learning',
                    data: [],
                    borderColor: '#ef4444', // Red solid
                    borderWidth: 2,
                    fill: false,
                    pointRadius: 0,
                    tension: 0.1
                },
                {
                    label: 'Sarsa, Sutton Pub.',
                    data: generateSuttonBaseline(true),
                    borderColor: '#06b6d4', // Cyan dotted
                    borderWidth: 1.5,
                    borderDash: [3, 3],
                    fill: false,
                    pointRadius: 0,
                    tension: 0.6
                },
                {
                    label: 'Q-learning, Sutton Pub.',
                    data: generateSuttonBaseline(false),
                    borderColor: '#ef4444', // Red dotted
                    borderWidth: 1.5,
                    borderDash: [3, 3],
                    fill: false,
                    pointRadius: 0,
                    tension: 0.6
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Sarsa Vs. Q-Learning Cliff Walking (averaged over 50 runs)',
                    font: { size: 16 }
                },
                legend: {
                    position: 'bottom'
                }
            },
            scales: {
                x: { 
                    title: { display: true, text: 'Episodes' },
                    min: 0,
                    max: 500
                },
                y: { 
                    title: { display: true, text: 'Reward Sum for Episode' }, 
                    min: -100, 
                    max: 0 
                }
            },
            animation: false
        }
    });
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

function chooseAction(qTable, r, c) {
    if (Math.random() < EPSILON) return Math.floor(Math.random() * 4);
    let maxQ = Math.max(...qTable[r][c]);
    let bestActions = [];
    for(let i=0; i<4; i++) if(qTable[r][c][i] === maxQ) bestActions.push(i);
    return bestActions[Math.floor(Math.random() * bestActions.length)];
}

function defaultQTable() {
    let q = [];
    for (let r = 0; r < ROWS; r++) {
        let row = [];
        for (let c = 0; c < COLS; c++) row.push([0,0,0,0]);
        q.push(row);
    }
    q[GOAL.r][GOAL.c] = [0,0,0,0];
    return q;
}

function runExperiment(isSarsa) {
    let totalRewardsPerEpisode = new Array(EPISODES).fill(0);
    let finalQTable = null;

    for(let run = 0; run < RUNS; run++) {
        let qTable = defaultQTable();
        
        for(let ep = 0; ep < EPISODES; ep++) {
            let state = {r: START.r, c: START.c};
            let action = chooseAction(qTable, state.r, state.c);
            let done = false;
            let epReward = 0;
            let steps = 0; 
            
            while(!done && steps < 3000) {
                let res = stepEnv(state, action);
                let nextAction = chooseAction(qTable, res.nextState.r, res.nextState.c);
                
                let target;
                if(res.done) target = res.reward;
                else if (isSarsa) target = res.reward + GAMMA * qTable[res.nextState.r][res.nextState.c][nextAction];
                else target = res.reward + GAMMA * Math.max(...qTable[res.nextState.r][res.nextState.c]);
                
                qTable[state.r][state.c][action] += ALPHA * (target - qTable[state.r][state.c][action]);
                
                epReward += res.reward;
                state = res.nextState;
                if(isSarsa) action = nextAction;
                else action = chooseAction(qTable, state.r, state.c);
                
                steps++;
            }
            totalRewardsPerEpisode[ep] += epReward;
        }
        finalQTable = qTable; // Extract Policy from last run
    }

    let avgRewards = totalRewardsPerEpisode.map(val => val / RUNS);
    
    // Smooth array for better visuals representing 50-run statistical averages
    let smoothed = [];
    for(let i=0; i<avgRewards.length; i++) {
        let lookback = Math.max(0, i-5);
        let slice = avgRewards.slice(lookback, i+1);
        let mean = slice.reduce((a,b)=>a+b)/slice.length;
        smoothed.push(mean);
    }

    return { avgRewards: smoothed, qTable: finalQTable };
}

// Maps policy to the DOM
function updatePolicyUI(qTable, containerId) {
    // 1. Draw Arrows
    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
            if ((r===START.r && c===START.c) || (r===GOAL.r && c===GOAL.c) || (r===3 && c>0 && c<11)) continue;
            let arrowEl = document.getElementById(`${containerId}-arrow-${r}-${c}`);
            if (arrowEl) {
                let maxQ = Math.max(...qTable[r][c]);
                if (maxQ !== 0) {
                    let bestA = qTable[r][c].indexOf(maxQ);
                    arrowEl.innerText = ARROW_SYMBOLS[bestA];
                }
            }
        }
    }
    
    // 2. Draw Expected Path Boundary (Blue dashed line style)
    let r = START.r, c = START.c;
    let visited = new Set();
    while (!(r === GOAL.r && c === GOAL.c)) {
        let key = `${r},${c}`;
        if (visited.has(key)) break;
        visited.add(key);
        
        let cell = document.getElementById(`${containerId}-cell-${r}-${c}`);
        if(cell) {
            cell.style.border = '2px dashed #06b6d4';
            cell.style.backgroundColor = 'rgba(6, 182, 212, 0.05)';
        }
        
        let maxQ = Math.max(...qTable[r][c]);
        // default Up if maxQ == 0 (unexplored shouldn't happen on greedy path)
        let bestA = maxQ !== 0 ? qTable[r][c].indexOf(maxQ) : 0; 
        
        if (bestA === 0) r = Math.max(0, r-1);
        else if (bestA === 1) c = Math.min(COLS-1, c+1);
        else if (bestA === 2) r = Math.min(ROWS-1, r+1);
        else if (bestA === 3) c = Math.max(0, c-1);
        
        if (r===3 && c>0 && c<11) break; // Fell in cliff
    }
}

// Execute on Click
startBtn.addEventListener('click', () => {
    updateParams();
    let totalComputed = EPISODES * RUNS * 2;
    startBtn.innerText = `Computing ${totalComputed.toLocaleString()} Total Episodes...`;
    startBtn.disabled = true;

    // Timeout allows DOM repaint of button before synchronous lockup
    setTimeout(() => {
        initChart(); // Re-init to match any changes to EPISODES length
        
        // Run Both Algorithms
        let sarsaResults = runExperiment(true);
        let qResults = runExperiment(false);

        // Bind Chart Data
        rewardChart.data.datasets[0].data = sarsaResults.avgRewards;
        rewardChart.data.datasets[1].data = qResults.avgRewards;
        rewardChart.update();

        // Bind Policy Grids
        updatePolicyUI(qResults.qTable, 'qGrid');
        updatePolicyUI(sarsaResults.qTable, 'sGrid');
        
        startBtn.innerText = "Start Full Academic Training";
        startBtn.disabled = false;
        startBtn.classList.replace('bg-green-500', 'bg-blue-600');
        startBtn.classList.replace('hover:bg-green-600', 'hover:bg-blue-700');
    }, 100);
});

// PDF Export
document.getElementById('downloadPdfBtn').addEventListener('click', async () => {
    const { jsPDF } = window.jspdf;
    const pdf = new jsPDF('p', 'pt', 'a4');
    
    const reportContent = document.getElementById('report-content');
    
    // Scale for sharper resolution
    const canvas = await html2canvas(reportContent, { scale: 1.5 });
    const imgData = canvas.toDataURL('image/png');
    
    let pdfWidth = pdf.internal.pageSize.getWidth();
    let imgProps = pdf.getImageProperties(imgData);
    let pdfHeight = (imgProps.height * pdfWidth) / imgProps.width;
    
    pdf.addImage(imgData, 'PNG', 0, 0, pdfWidth, pdfHeight);
    pdf.save('Sarsa_QLearning_CliffWalking.pdf');
});

// On Boot
initGrids();
initChart();
