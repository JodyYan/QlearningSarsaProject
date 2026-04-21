// rl_test.js: Headless validation of RL logic for Cliff Walking
// To test epsilon=0.1, alpha=0.5, gamma=1.0
const ROWS = 4;
const COLS = 12;
const START = {r: 3, c: 0};
const GOAL = {r: 3, c: 11};

const ACTIONS = [0, 1, 2, 3]; // 0: Up, 1: Right, 2: Down, 3: Left
const EPSILON = 0.1;
const ALPHA = 0.5;
const GAMMA = 1.0;
const EPISODES = 500;
const RUNS = 50;

function step(state, action) {
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
        return { nextState: {r,c}, reward: 0, done: true }; 
        // With gamma=1.0, to match sutton exactly, goal transition gives 0
        // Wait, standard is -1 per step, what if goal transition is -1?
        // Let's use -1 for goal transition as well, as it's a step.
    }
    
    return { nextState: {r,c}, reward: -1, done: false };
}

function chooseAction(qTable, r, c) {
    if (Math.random() < EPSILON) {
        return Math.floor(Math.random() * 4);
    }
    let maxQ = Math.max(...qTable[r][c]);
    let bestActions = [];
    for (let a = 0; a < 4; a++) if (qTable[r][c][a] === maxQ) bestActions.push(a);
    return bestActions[Math.floor(Math.random() * bestActions.length)];
}

function initQTable() {
    let q = [];
    for (let r = 0; r < ROWS; r++) {
        let row = [];
        for (let c = 0; c < COLS; c++) {
            row.push([0,0,0,0]);
        }
        q.push(row);
    }
    // Goal is terminal, value is 0
    q[GOAL.r][GOAL.c] = [0,0,0,0];
    return q;
}

function runExperiment(isSarsa) {
    let totalRewardsPerEpisode = new Array(EPISODES).fill(0);
    let finalQTable = null;

    for (let run = 0; run < RUNS; run++) {
        let qTable = initQTable();
        
        for (let ep = 0; ep < EPISODES; ep++) {
            let state = {r: START.r, c: START.c};
            let action = chooseAction(qTable, state.r, state.c);
            let done = false;
            let totalReward = 0;
            let stepCount = 0;
            
            while (!done && stepCount < 2000) { // Limit steps
                let res = step(state, action);
                let nextAction = chooseAction(qTable, res.nextState.r, res.nextState.c);
                
                let target;
                if (res.done) {
                    target = res.reward;
                } else if (isSarsa) {
                    target = res.reward + GAMMA * qTable[res.nextState.r][res.nextState.c][nextAction];
                } else {
                    let maxQNext = Math.max(...qTable[res.nextState.r][res.nextState.c]);
                    target = res.reward + GAMMA * maxQNext;
                }
                
                qTable[state.r][state.c][action] += ALPHA * (target - qTable[state.r][state.c][action]);
                
                totalReward += res.reward;
                state = res.nextState;
                if (isSarsa) {
                    action = nextAction;
                } else {
                    action = chooseAction(qTable, state.r, state.c);
                }
                stepCount++;
            }
            totalRewardsPerEpisode[ep] += totalReward;
        }
        finalQTable = qTable; // Keep last run's table
    }

    let avgRewards = totalRewardsPerEpisode.map(val => val / RUNS);
    return { avgRewards, finalQTable };
}

console.log("Running SARSA...");
let sarsaRes = runExperiment(true);

console.log("Running Q-learning...");
let qRes = runExperiment(false);

// Calculate final average over last 50 episodes
let last50_sarsa = sarsaRes.avgRewards.slice(-50).reduce((a,b)=>a+b)/50;
let last50_q = qRes.avgRewards.slice(-50).reduce((a,b)=>a+b)/50;

console.log(`SARSA final average reward (last 50 eps): ${last50_sarsa.toFixed(2)}`);
console.log(`Q-learning final average reward (last 50 eps): ${last50_q.toFixed(2)}`);

function getPathLenAndReward(qTable) {
    let r = START.r, c = START.c;
    let path = [];
    let visited = new Set();
    let totalR = 0;
    while (!(r === GOAL.r && c === GOAL.c)) {
        let key = `${r},${c}`;
        if (visited.has(key)) break;
        visited.add(key);
        path.push([r,c]);
        
        let maxQ = Math.max(...qTable[r][c]);
        let bestA = qTable[r][c].indexOf(maxQ);
        if (bestA === 0) r = Math.max(0, r-1);
        else if (bestA === 1) c = Math.min(COLS-1, c+1);
        else if (bestA === 2) r = Math.min(ROWS-1, r+1);
        else if (bestA === 3) c = Math.max(0, c-1);
        
        if (r===3 && c>0 && c<11) {
            totalR -= 100;
            break;
        } else {
            totalR -= 1;
        }
    }
    return {path: path.length, reward: totalR};
}

let sp = getPathLenAndReward(sarsaRes.finalQTable);
let qp = getPathLenAndReward(qRes.finalQTable);

console.log(`SARSA strict greedy path reward: ${sp.reward}`);
console.log(`Q-learning strict greedy path reward: ${qp.reward}`);
