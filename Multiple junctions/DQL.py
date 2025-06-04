import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

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

# DQL parameters
TOTAL_STEPS = 10000
BATCH_SIZE = 32
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.001
MEMORY_SIZE = 10000
UPDATE_TARGET_FREQUENCY = 10  # Update target network every N steps
ACTIONS = [0, 1]  # 0: keep phase, 1: switch phase
MIN_GREEN_STEPS = 5
last_switch_step = {jid: -MIN_GREEN_STEPS for jid in junctions}

# Step 5: Helper functions for traffic state
def get_queue_length(detectors):
    return sum([traci.lanearea.getLastStepVehicleNumber(d) for d in detectors])

def get_current_phase(jid):
    return traci.trafficlight.getPhase(jid)

def get_state_array():
    """Returns state as a numpy array for neural network input"""
    state = []
    for jid in junctions:
        # Add queue length (normalized by dividing by 20 to keep values in reasonable range)
        state.append(get_queue_length(junctions[jid]) / 20.0)
        
        # Add phase as one-hot encoding
        program = traci.trafficlight.getAllProgramLogics(jid)[0]
        num_phases = len(program.phases)
        current_phase = get_current_phase(jid)
        one_hot = np.zeros(num_phases)
        one_hot[current_phase] = 1
        state.extend(one_hot)
    
    return np.array(state, dtype=np.float32)

def get_reward(state_array):
    """Calculate reward based on queue lengths"""
    # Extract queue lengths from state array (every Nth element where N depends on junction phase counts)
    queue_lengths = []
    index = 0
    for jid in junctions:
        queue_lengths.append(state_array[index] * 20.0)  # Un-normalize
        program = traci.trafficlight.getAllProgramLogics(jid)[0]
        index += 1 + len(program.phases)  # Skip queue + one-hot phases
    
    return -float(sum(queue_lengths))

def apply_action(action, step):
    if action == 0:
        return
    for jid in junctions:
        if step - last_switch_step[jid] >= MIN_GREEN_STEPS:
            program = traci.trafficlight.getAllProgramLogics(jid)[0]
            next_phase = (get_current_phase(jid) + 1) % len(program.phases)
            traci.trafficlight.setPhase(jid, next_phase)
            last_switch_step[jid] = step

# Step 6: Deep Q-Network Implementation
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON_START
        self.gamma = GAMMA
        self.learning_rate = LEARNING_RATE
        
        # Main model (trained every step)
        self.model = self._build_model()
        
        # Target model (periodically updated)
        self.target_model = self._build_model()
        self.update_target_model()
        
    def _build_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        """Copy weights from main model to target model"""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        act_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        """Train model using experience replay"""
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        
        states = np.zeros((batch_size, self.state_size))
        targets = np.zeros((batch_size, self.action_size))
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = self.model.predict(state.reshape(1, -1), verbose=0)[0]
            
            if done:
                target[action] = reward
            else:
                t = self.target_model.predict(next_state.reshape(1, -1), verbose=0)[0]
                target[action] = reward + self.gamma * np.amax(t)
            
            states[i] = state
            targets[i] = target
        
        self.model.fit(states, targets, epochs=1, verbose=0)
        
        if self.epsilon > EPSILON_END:
            self.epsilon *= EPSILON_DECAY

# Step 7: Calculate state size dynamically
def get_state_size():
    """Calculate the total size of the state vector"""
    size = 0
    for jid in junctions:
        program = traci.trafficlight.getAllProgramLogics(jid)[0]
        size += 1  # Queue length
        size += len(program.phases)  # One-hot encoding of phases
    return size

# Initialize agent
state_size = get_state_size()
action_size = len(ACTIONS)
agent = DQNAgent(state_size, action_size)

# Step 8: Training Loop
step_history = []
reward_history = []
queue_history = []
cumulative_reward = 0
done = False

print("\n=== Starting DQL Training ===")
for step in range(TOTAL_STEPS):
    state = get_state_array()
    action = agent.act(state)
    apply_action(action, step)
    traci.simulationStep()
    
    next_state = get_state_array()
    reward = get_reward(next_state)
    cumulative_reward += reward
    
    # Store experience in replay memory
    agent.remember(state, action, reward, next_state, done)
    
    # Train the model with batch of experiences
    agent.replay(BATCH_SIZE)
    
    # Update target model periodically
    if step % UPDATE_TARGET_FREQUENCY == 0:
        agent.update_target_model()
    
    if step % 100 == 0:
        print(f"Step {step}, Reward: {reward:.2f}, Cumulative: {cumulative_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
        step_history.append(step)
        reward_history.append(cumulative_reward)
        
        # Calculate total queue length for monitoring
        total_queue = 0
        for jid in junctions:
            total_queue += get_queue_length(junctions[jid])
        queue_history.append(total_queue)

# Step 9: Save the trained model
agent.model.save("traffic_dqn_model.h5")
print("Model saved as traffic_dqn_model.h5")

# Step 10: Cleanup and Visualization
traci.close()

plt.figure(figsize=(10, 6))
plt.plot(step_history, reward_history, label="Cumulative Reward")
plt.xlabel("Steps")
plt.ylabel("Cumulative Reward")
plt.title("DQL: Cumulative Reward Over Time")
plt.legend()
plt.grid(True)
plt.savefig("dql_reward_plot.png")
plt.show()
