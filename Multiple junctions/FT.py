import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Ensure SUMO environment is correctly set
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci

# Step 2: Define SUMO configuration
Sumo_config = [
    'sumo-gui',
    '-c', 'nairobi.sumocfg',
    '--step-length', '0.1'
]

# Step 3: Open SUMO connection
traci.start(Sumo_config)
traci.gui.setSchema("View #0", "real world")

# Step 4: Define junctions and their detectors
junctions = {
    "J8": ["MOI023", "MOI024", "MOI025", "MOI026", "e2_6", "e2_7"],
    "296111410": ["MOI019", "MOI020", "e2_0", "e2_1", "MOI021", "MOI022"],
    "10961054607": ["MOI015", "MOI016", "e2_3", "e2_4", "MOI017", "MOI018", "e2_2"],
    "2875606102": ["MOI011", "MOI012", "MOI013", "MOI014"],
    "30986871": ["MOI07", "MOI08", "SUM02", "SUM03", "MOI09", "MOI010"],
    "10873342297": ["MOI03", "MOI04", "MOI05", "MOI016", "SUM1"],
    "6364594184": ["R01", "HS04", "HS05", "HS06", "MOI01", "MOI02", "HS01", "HS02", "HS03"],
    "6370188407": ["HS07", "HS08", "HS09", "e2_13", "HS010", "HS011", "HS012"]
}

# RL parameters
TOTAL_STEPS = 10000
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.1
ACTIONS = [0, 1]  # 0: keep phase, 1: switch phase
MIN_GREEN_STEPS = 5
last_switch_step = {jid: -MIN_GREEN_STEPS for jid in junctions}
Q_table = {}

# Step 5: Helper functions
def get_queue_length(detectors):
    return sum([traci.lanearea.getLastStepVehicleNumber(d) for d in detectors])

def get_current_phase(jid):
    return traci.trafficlight.getPhase(jid)

def get_state():
    state = []
    for jid in junctions:
        state.append(get_queue_length(junctions[jid]))
        state.append(get_current_phase(jid))
    return tuple(state)

def get_reward(state):
    queue_lengths = state[::2]  # Every other element starting at 0 is a queue length
    return -float(sum(queue_lengths))

def get_max_Q(s):
    if s not in Q_table:
        Q_table[s] = np.zeros(len(ACTIONS))
    return np.max(Q_table[s])

def update_Q(old_state, action, reward, new_state):
    if old_state not in Q_table:
        Q_table[old_state] = np.zeros(len(ACTIONS))
    old_q = Q_table[old_state][action]
    best_future_q = get_max_Q(new_state)
    Q_table[old_state][action] = old_q + ALPHA * (reward + GAMMA * best_future_q - old_q)

def get_action(state):
    if random.random() < EPSILON:
        return random.choice(ACTIONS)
    if state not in Q_table:
        Q_table[state] = np.zeros(len(ACTIONS))
    return np.argmax(Q_table[state])

def apply_action(action, step):
    if action == 0:
        return
    for jid in junctions:
        if step - last_switch_step[jid] >= MIN_GREEN_STEPS:
            program = traci.trafficlight.getAllProgramLogics(jid)[0]
            next_phase = (get_current_phase(jid) + 1) % len(program.phases)
            traci.trafficlight.setPhase(jid, next_phase)
            last_switch_step[jid] = step

# Step 6: Training Loop
step_history = []
reward_history = []
queue_history = []
cumulative_reward = 0

print("\n=== Starting Training ===")
for step in range(TOTAL_STEPS):
    state = get_state()
    action = get_action(state)
    apply_action(action, step)
    traci.simulationStep()
    new_state = get_state()
    reward = get_reward(new_state)
    cumulative_reward += reward
    update_Q(state, action, reward, new_state)

    if step % 100 == 0:
        print(f"Step {step}, Reward: {reward:.2f}, Cumulative: {cumulative_reward:.2f}")
        step_history.append(step)
        reward_history.append(cumulative_reward)
        queue_history.append(sum(new_state[::2]))

# Step 7: Cleanup and Visualization
traci.close()

plt.figure(figsize=(10, 6))
plt.plot(step_history, reward_history, label="Cumulative Reward")
plt.xlabel("Steps")
plt.ylabel("Cumulative Reward")
plt.title("Cumulative Reward Over Time")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(step_history, queue_history, label="Queue Length")
plt.xlabel("Steps")
plt.ylabel("Total Queue Length")
plt.title("Queue Length Over Time")
plt.legend()
plt.grid(True)
plt.show()
