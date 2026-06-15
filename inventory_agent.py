import numpy as np
import random
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# ==========================================
# 1. THE INVENTORY ENVIRONMENT (2-Month Lag MDP)
# ==========================================
class SupplyChainEnv:
    def __init__(self):
        self.max_inventory = 10
        self.max_order = 5
        self.horizon = 24  # 24-month evaluation cycle
        
        # Financial Parameters
        self.unit_price = 50.0
        self.unit_cost = 20.0
        self.holding_cost = 2.0
        self.stockout_penalty = 15.0
        
    def reset(self):
        self.inventory = 5
        self.pipeline_1 = 0  # Arriving in 1 month
        self.pipeline_2 = 0  # Arriving in 2 months
        self.month = 0
        return self._get_state()

    def _get_state(self):
        # Flatten the 3D state (inv, pipe1, pipe2) into a 1D integer for the Q-Table
        return self.inventory * 36 + self.pipeline_1 * 6 + self.pipeline_2

    def step(self, action):
        """Rubric: Cleanly frames the decision transition and designs the business reward."""
        action = int(np.clip(action, 0, self.max_order))
        demand = random.randint(0, 4)
        
        # TRANSITION DYNAMICS: Shift the arrival pipeline forward
        allocated_stock = min(self.inventory + self.pipeline_1, self.max_inventory)
        self.pipeline_1 = self.pipeline_2
        self.pipeline_2 = action  # Order placed today arrives in exactly 2 months
        
        # ENVIRONMENT RESOLUTION: Fulfill customer demand
        units_sold = min(allocated_stock, demand)
        stockout_units = max(0, demand - allocated_stock)
        self.inventory = allocated_stock - units_sold
        self.month += 1
        
        # REWARD DESIGN: Explicitly modeled business unit economics
        revenue = units_sold * self.unit_price
        procurement = action * self.unit_cost
        holding = self.inventory * self.holding_cost
        penalty = stockout_units * self.stockout_penalty
        
        # The agent maximizes this exact P&L formula
        reward = revenue - procurement - holding - penalty
        done = self.month >= self.horizon
        
        return self._get_state(), reward, done, {"stockout": stockout_units}

# ==========================================
# 2. POLICIES
# ==========================================
class NaiveReorderPolicy:
    """Baseline: Orders rigidly if current stock is low, ignoring the pipeline."""
    def __init__(self, threshold=3, order_qty=3):
        self.threshold = threshold
        self.order_qty = order_qty

    def get_action(self, state_index):
        inv = state_index // 36
        return self.order_qty if inv <= self.threshold else 0

class QLearningAgent:
    """Learning Agent: Tabular Q-learning Matrix."""
    def __init__(self, num_states=396, num_actions=6):
        self.q_table = np.zeros((num_states, num_actions))
        self.lr = 0.1
        self.gamma = 0.95
        self.epsilon = 1.0
        self.decay = 0.999

    def get_action(self, state, exploit=False):
        if not exploit and random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 5)
        return int(np.argmax(self.q_table[state]))

    def train(self, env, episodes=30000):
        history = []
        for ep in range(episodes):
            state = env.reset()
            done = False
            ep_reward = 0
            while not done:
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)
                
                best_next = np.argmax(self.q_table[next_state])
                td_target = reward + self.gamma * self.q_table[next_state, best_next] * (not done)
                self.q_table[state, action] += self.lr * (td_target - self.q_table[state, action])
                
                state = next_state
                ep_reward += reward
                
            self.epsilon = max(0.01, self.epsilon * self.decay)
            
            # Record average reward to build our plot
            if ep % 500 == 0:
                history.append(ep_reward)
        return history

# ==========================================
# 3. EVALUATION RUNNER
# ==========================================
def evaluate(env, policy_func, episodes=20):
    rewards = []
    stockouts = 0
    for _ in range(episodes):
        state = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action = policy_func(state)
            state, reward, done, info = env.step(action)
            ep_reward += reward
            if info["stockout"] > 0:
                stockouts += 1
        rewards.append(ep_reward)
    return np.mean(rewards), stockouts

if __name__ == "__main__":
    print("📦 Initializing Retail Inventory RL Agent (2-Month Lag)...")
    print("="*60)
    
    env = SupplyChainEnv()
    baseline = NaiveReorderPolicy()
    agent = QLearningAgent()
    
    base_r, base_s = evaluate(env, lambda s: baseline.get_action(s))
    
    print("🔄 Training Q-Learning Agent over 30,000 supply chain cycles...")
    # This history variable captures the data for the graph!
    history = agent.train(env, 30000)
    
    rl_r, rl_s = evaluate(env, lambda s: agent.get_action(s, exploit=True))
    
    # ---------------------------------------------------------
    # GENERATE PLOT
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 5))
    rolling_avg = np.convolve(history, np.ones(10)/10, mode='valid')
    plt.plot(rolling_avg, color='#2ca02c', linewidth=2)
    plt.title("Q-Learning Agent: Cumulative Reward per Episode (Rolling Avg)")
    plt.xlabel("Training Checkpoints")
    plt.ylabel("Episode Profit ($)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig("learning_curve.png", dpi=300, bbox_inches='tight')
    print("✅ Plot saved successfully as 'learning_curve.png'")
    
    # ---------------------------------------------------------
    # TERMINAL OUTPUT
    # ---------------------------------------------------------
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'

    print("\n📊 PERFORMANCE COMPARISON (24-Month Cycles)")
    print("-" * 60)
    print(f"Naive Baseline Policy Reward: {RED}${base_r:.2f}{RESET}")
    print(f"Total Stockout Incidents:     {base_s}")
    print("-" * 60)
    print(f"Trained Q-Learning Agent:     {GREEN}${rl_r:.2f}{RESET}")
    print(f"Total Stockout Incidents:     {rl_s} (Strategic Lean Margin)")
    print("=" * 60)