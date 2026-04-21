const ROWS = 4;
const COLS = 12;
const START = {r: 3, c: 0};
const GOAL = {r: 3, c: 11};

const ACTIONS = [0, 1, 2, 3]; // 0: Up, 1: Right, 2: Down, 3: Left
const EPSILON = 0.1;
const ALPHA = 0.5;
const GAMMA = 1.0;
const EPISODES = 500;

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
        return { nextState: {r,c}, reward: -1, done: true }; 
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
            row.push([0,0,0,0]); // Should we initialize to 0 or highly negative? 0 is standard.
        }
        q.push(row);
    }
    q[GOAL.r][GOAL.c] = [0,0,0,0];
    return q;
}

function runExperiment(isSarsa) {
    let qTable = initQTable();
    let rewards = [];
    
    for (let ep = 0; ep < EPISODES; ep++) {
        let state = {r: START.r, c: START.c};
        let action = chooseAction(qTable, state.r, state.c);
        let done = false;
        let totalReward = 0;
        let stepCount = 0;
        
        while (!done && stepCount < 5000) { 
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
        rewards.push(totalReward);
    }
    return rewards;
}

let sr = runExperiment(true);
let qr = runExperiment(false);
console.log("SARSA 500th EP:", sr[499]);
console.log("Q 500th EP:", qr[499]);
