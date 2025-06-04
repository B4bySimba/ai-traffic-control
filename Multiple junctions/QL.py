import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt


if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci

Sumo_config = [
    'sumo-gui',
    '-c', 'nairobi.sumocfg',
    '--step-length', '0.10',
    '--delay', '1000',
    '--lateral-resolution', '0'
]

traci.start(Sumo_config)
traci.gui.setSchema("View #0", "real world")


# Define mapping between junctions and their detectors
junction_detectors = {
    "J8": ["MOI023", "MOI024", "MOI025", "MOI026", "e2_6", "e2_7"],
    "296111410": ["MOI019", "MOI020", "e2_0", "e2_1", "MOI021", "MOI022"],
    "10961054607": ["MOI015", "MOI016", "e2_3", "e2_4", "MOI017", "MOI018", "e2_2"],
    "2875606102": ["MOI011", "MOI012", "MOI013", "MOI014"],
    "30986871": ["MOI07", "MOI08", "SUM02", "SUM03", "MOI09", "MOI010"],
    "10873342297": ["MOI03", "MOI04", "MOI05", "MOI016", "SUM1"],
    "6364594184": ["R01", "HS04", "HS05", "HS06", "MOI01", "MOI02", "HS01", "HS02", "HS03"],
    "6370188407": ["HS07", "HS08", "HS09", "e2_13", "HS010", "HS011", "HS012"]
}

# Dictionary to store current queue lengths for each detector
queue_lengths = {}

# Dictionary to store current phase for each junction
current_phases = {}

# Initialize queue lengths for all detectors
for junction, detectors in junction_detectors.items():
    for detector in detectors:
        queue_lengths[detector] = 0
    current_phases[junction] = 0

# Dictionary to track last switch step for each junction
last_switch_steps = {}
for junction in junction_detectors:
    last_switch_steps[junction] = -100

# ---- Reinforcement Learning Hyperparameters ----
TOTAL_STEPS = 10000    # The total number of simulation steps for continuous (online) training.

ALPHA = 0.1            # Learning rate (α) between[0, 1]
GAMMA = 0.9            # Discount factor (γ) between[0, 1]
EPSILON = 0.1          # Exploration rate (ε) between[0, 1]

ACTIONS = [0, 1]       # The discrete action space (0 = keep phase, 1 = switch phase)

# Separate Q-table for each junction
Q_tables = {}
for junction in junction_detectors:
    Q_tables[junction] = {}

# ---- Additional Stability Parameters ----
MIN_GREEN_STEPS = 100

# -------------------------
# Step 7: Define Functions
# -------------------------

def get_max_Q_value_of_state(junction, s):
    """Get max Q-value for a state in a specific junction's Q-table"""
    if s not in Q_tables[junction]:
        Q_tables[junction][s] = np.zeros(len(ACTIONS))
    return np.max(Q_tables[junction][s])

def get_reward(junction, state):
    """
    Simple reward function:
    Negative of total queue length to encourage shorter queues.
    """
    # Extract queue lengths from state (all elements except the last one, which is current_phase)
    queue_values = state[:-1]
    total_queue = sum(queue_values)
    reward = -float(total_queue)
    return reward

def get_state(junction):
    """Get the current state for a specific junction"""
    detector_values = []
    
    # Get queue lengths for all detectors of this junction
    for detector in junction_detectors[junction]:
        detector_values.append(get_queue_length(detector))
    
    # Append current phase
    detector_values.append(current_phases[junction])
    
    # Return as tuple (immutable, can be used as dictionary key)
    return tuple(detector_values)

def apply_action(junction, action):
    """
    Executes the chosen action on the traffic light for a specific junction
    """
    global current_simulation_step
    
    if action == 0:
        # Do nothing (keep current phase)
        return
    
    elif action == 1:
        # Check if minimum green time has passed before switching
        if current_simulation_step - last_switch_steps[junction] >= MIN_GREEN_STEPS:
            try:
                program = traci.trafficlight.getAllProgramLogics(junction)[0]
                num_phases = len(program.phases)
                next_phase = (current_phases[junction] + 1) % num_phases
                traci.trafficlight.setPhase(junction, next_phase)
                last_switch_steps[junction] = current_simulation_step
                current_phases[junction] = next_phase
            except:
                # If junction is not a traffic light, just skip
                print(f"Warning: Junction {junction} might not be a traffic light or doesn't exist.")

def update_Q_table(junction, old_state, action, reward, new_state):
    """Update Q-table for a specific junction"""
    if old_state not in Q_tables[junction]:
        Q_tables[junction][old_state] = np.zeros(len(ACTIONS))
    
    # 1) Predict current Q-values from old_state (current state)
    old_q = Q_tables[junction][old_state][action]
    # 2) Predict Q-values for new_state to get max future Q (new state)
    best_future_q = get_max_Q_value_of_state(junction, new_state)
    # 3) Incorporate ALPHA to partially update the Q-value and update Q table
    Q_tables[junction][old_state][action] = old_q + ALPHA * (reward + GAMMA * best_future_q - old_q)

def get_action_from_policy(junction, state):
    """Select action using epsilon-greedy policy for a specific junction"""
    if random.random() < EPSILON:
        return random.choice(ACTIONS)
    else:
        if state not in Q_tables[junction]:
            Q_tables[junction][state] = np.zeros(len(ACTIONS))
        return int(np.argmax(Q_tables[junction][state]))

def get_queue_length(detector_id):
    """Get queue length from detector"""
    try:
        return traci.lanearea.getLastStepVehicleNumber(detector_id)
    except:
        # If detector doesn't exist, return 0
        print(f"Warning: Detector {detector_id} might not exist.")
        return 0

def get_current_phase(tls_id):
    """Get current phase of traffic light"""
    try:
        return traci.trafficlight.getPhase(tls_id)
    except:
        # If traffic light doesn't exist, return 0
        print(f"Warning: Traffic light {tls_id} might not exist.")
        return 0

def update_junction_phases():
    """Update the current phases for all junctions"""
    for junction in junction_detectors:
        try:
            current_phases[junction] = get_current_phase(junction)
        except:
            # Skip if not a traffic light
            pass

# -------------------------
# Step 8: Fully Online Continuous Learning Loop
# -------------------------

# Dictionaries to record data for plotting
step_history = []
reward_history = {junction: [] for junction in junction_detectors}
queue_history = {junction: [] for junction in junction_detectors}
cumulative_rewards = {junction: 0.0 for junction in junction_detectors}

print("\n=== Starting Fully Online Continuous Learning for Multiple Junctions ===")
for step in range(TOTAL_STEPS):
    current_simulation_step = step
    
    # Update phases at the beginning of each step
    update_junction_phases()
    
    # Store old states for all junctions
    old_states = {}
    actions = {}
    
    # First, get states and decide actions for all junctions
    for junction in junction_detectors:
        old_states[junction] = get_state(junction)
        actions[junction] = get_action_from_policy(junction, old_states[junction])
    
    # Then, apply all actions
    for junction in junction_detectors:
        apply_action(junction, actions[junction])
    
    # Advance simulation by one step
    traci.simulationStep()
    
    # Get new states and update Q-tables for all junctions
    for junction in junction_detectors:
        new_state = get_state(junction)
        reward = get_reward(junction, new_state)
        cumulative_rewards[junction] += reward
        
        update_Q_table(junction, old_states[junction], actions[junction], reward, new_state)
        
        # Print detailed info every 100 steps
        if step % 100 == 0:
            print(f"Step {step}, Junction: {junction}")
            print(f"  State: {old_states[junction]}")
            print(f"  Action: {actions[junction]}")
            print(f"  New State: {new_state}")
            print(f"  Reward: {reward:.2f}")
            print(f"  Cumulative Reward: {cumulative_rewards[junction]:.2f}")
    
    # Record data every 10 steps to reduce overhead
    if step % 10 == 0:
        step_history.append(step)
        for junction in junction_detectors:
            reward_history[junction].append(cumulative_rewards[junction])
            queue_history[junction].append(sum(get_state(junction)[:-1]))  # Sum of queue lengths

# -------------------------
# Step 9: Close connection between SUMO and Traci
# -------------------------
traci.close()

# Print final Q-table info
print("\nOnline Training completed.")
for junction in junction_detectors:
    print(f"\nJunction: {junction}, Final Q-table size: {len(Q_tables[junction])}")
    # Print a few sample entries to avoid overwhelming console
    sample_count = min(5, len(Q_tables[junction]))
    print(f"Sample Q-table entries (showing {sample_count}):")
    for i, (state, actions) in enumerate(list(Q_tables[junction].items())[:sample_count]):
        print(f"  State: {state} -> Q-values: {actions}")

# -------------------------
# Visualization of Results
# -------------------------

# Plot Cumulative Reward over Simulation Steps for each junction
plt.figure(figsize=(12, 8))
for junction in junction_detectors:
    plt.plot(step_history, reward_history[junction], linestyle='-', label=f"Junction {junction}")

plt.xlabel("Simulation Step")
plt.ylabel("Cumulative Reward")
plt.title("RL Training: Cumulative Reward over Steps")
plt.legend()
plt.grid(True)
plt.savefig("cumulative_rewards.png")
plt.show()

# Plot Total Queue Length over Simulation Steps for each junction
plt.figure(figsize=(12, 8))
for junction in junction_detectors:
    plt.plot(step_history, queue_history[junction], linestyle='-', label=f"Junction {junction}")

plt.xlabel("Simulation Step")
plt.ylabel("Total Queue Length")
plt.title("RL Training: Queue Length over Steps")
plt.legend()
plt.grid(True)
plt.savefig("queue_lengths.png")
plt.show()