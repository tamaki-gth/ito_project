import numpy as np
import matplotlib.pyplot as plt

from EOM import RungeKutta, m1, m2, l1, l2, p1, p2, J1, J2, g, tau

# =========================
# ハイパーパラメータ
# =========================
GAMMA = 0.9   # 割引率
ALPHA = 0.1   # 学習率
DT = tau      # 1ステップの時間刻み [s]
MAX_STEPS = 2000      # 1エピソードのステップ数
NUM_EPISODES = 5000   # エピソード数
SCALE_CE = 0.01       # 累積消費エネルギーのスケール（要調整）

# =========================
# 手先速度の計算
# =========================
def forward_kinematics_and_velocity(z):
    theta1, theta1_dot, theta2, theta2_dot = z

    # 手先位置
    x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
    y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)

    # ヤコビアン
    dx_dtheta1 = -l1 * np.sin(theta1) - l2 * np.sin(theta1 + theta2)
    dx_dtheta2 = -l2 * np.sin(theta1 + theta2)
    dy_dtheta1 =  l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
    dy_dtheta2 =  l2 * np.cos(theta1 + theta2)

    x_dot = dx_dtheta1 * theta1_dot + dx_dtheta2 * theta2_dot
    y_dot = dy_dtheta1 * theta1_dot + dy_dtheta2 * theta2_dot

    v = np.sqrt(x_dot**2 + y_dot**2)
    return x, y, x_dot, y_dot, v

# =========================
# 状態の離散化
# =========================
class State:
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.num_states = 4**4

        self.q_table = np.random.uniform(low=-1, high=1,
                                         size=(self.num_states, num_actions))

        # 4分割用の閾値
        self.theta1_bins    = np.linspace(-np.pi, np.pi, 5)[1:-1]
        self.theta1dot_bins = np.linspace(-10.0, 10.0, 5)[1:-1]
        self.theta2_bins    = np.linspace(0.0, np.deg2rad(150), 5)[1:-1]
        self.theta2dot_bins = np.linspace(-10.0, 10.0, 5)[1:-1]

    def analog2digitize(self, observation):
        theta1, theta1_dot, theta2, theta2_dot = observation

        i1 = np.digitize(theta1,     self.theta1_bins)
        i2 = np.digitize(theta1_dot, self.theta1dot_bins)
        i3 = np.digitize(theta2,     self.theta2_bins)
        i4 = np.digitize(theta2_dot, self.theta2dot_bins)

        # 4×4×4×4 = 256状態
        state = i1 + 4*i2 + 4*4*i3 + 4*4*4*i4
        return state

    def update_Q_table(self, observation, action, reward, observation_next):
        s = self.analog2digitize(observation)
        s_next = self.analog2digitize(observation_next)
        max_q_next = np.max(self.q_table[s_next, :])

        self.q_table[s, action] += ALPHA * (reward + GAMMA * max_q_next - self.q_table[s, action])

    def decide_action(self, observation, episode):
        state = self.analog2digitize(observation)
        epsilon = 0.5 * (1.0 / (episode + 1))

        if np.random.uniform(0, 1) < epsilon:
            action = np.random.choice(self.num_actions)
        else:
            action = np.argmax(self.q_table[state, :])
        return action

class Agent:
    def __init__(self, num_actions):
        self.state = State(num_actions)

    def update_Q_function(self, observation, action, reward, observation_next):
        self.state.update_Q_table(observation, action, reward, observation_next)

    def get_action(self, observation, episode):
        return self.state.decide_action(observation, episode)

# =========================
# 環境
# =========================
class Environment:
    def __init__(self):
        self.num_actions = 9
        self.agent = Agent(self.num_actions)

        self.tau1_mag = 2.0   # トルクの大きさ（EOMのスケールに合わせて調整）
        self.tau2_mag = 2.0

        self.reward_history = []
        self.eval_history = []

        self.reset()

    def reset(self):
        self.z = np.array([0.0, 0.0, 0.0, 0.0])
        return self.z

    def decode_action(self, action_index):
        tau1_set = [-self.tau1_mag, 0.0, self.tau1_mag]
        tau2_set = [-self.tau2_mag, 0.0, self.tau2_mag]

        i1 = action_index % 3
        i2 = action_index // 3

        tau1 = tau1_set[i1]
        tau2 = tau2_set[i2]
        return tau1, tau2

    def step(self, action_index):
        tau1, tau2 = self.decode_action(action_index)

        theta1, theta1_dot, theta2, theta2_dot = self.z

        ce1 = abs(tau1 * theta1_dot) * DT
        ce2 = abs(tau2 * theta2_dot) * DT
        ce_step = ce1 + ce2

        # 力学で1ステップ進める
        self.z = RungeKutta(self.z, DT, m1, m2, l1, l2, p1, p2, J1, J2, tau1, tau2, g)
        next_state = self.z.copy()

        reward = -SCALE_CE * ce_step
        done = False

        return next_state, reward, done, ce_step

    def angle_reward(self,theta_a_deg):
      if 0 <= theta_a_deg < 45:
        return theta_a_deg / 45
      elif 45 <= theta_a_deg < 90:
        return (90 - theta_a_deg) / 45
      elif 90 <= theta_a_deg < 135:
        return (theta_a_deg - 90) / 45
      elif 135 <= theta_a_deg <= 180:
        return (180 - theta_a_deg) / 45
      else:
        return 0

    def compute_release_reward(self, z, ce_total):
        theta1, theta1_dot, theta2, theta2_dot = z

        theta_a = theta1 + theta2 - np.pi/2
        _, _, _, _, v2 = forward_kinematics_and_velocity(z)

        theta_a_deg=np.rad2deg(theta_a)
        reward_final = self.angle_reward(theta_a_deg) * v2 - SCALE_CE * ce_total
        return reward_final

    def run(self):
        for episode in range(NUM_EPISODES):
            observation = self.reset()
            ce_total = 0.0
            total_reward = 0.0

            for step in range(MAX_STEPS):
                action = self.agent.get_action(observation, episode)
                observation_next, reward_step, done, ce_step = self.step(action)

                ce_total += ce_step
                total_reward += reward_step

                self.agent.update_Q_function(observation, action, reward_step, observation_next)
                observation = observation_next

                if done:
                    break

            reward_final = self.compute_release_reward(self.z, ce_total)
            total_reward += reward_final

            # 終端状態での近似的なQ更新（同じ状態にとどまると仮定）
            final_state = self.z.copy()
            final_action = self.agent.get_action(final_state, episode)
            self.agent.update_Q_function(final_state, final_action, reward_final, final_state)

            self.reward_history.append(total_reward)

            # greedy 評価
#            eval_return = self.evaluate_policy(num_eval_episodes=3)
 #           self.eval_history.append(eval_return)

#            if (episode + 1) % 100 == 0:
#                print(f"Episode {episode+1}, total_reward = {total_reward:.3f}, eval = {eval_return:.3f}")

    '''def evaluate_policy(self, num_eval_episodes=3):
        returns = []
        for _ in range(num_eval_episodes):
            observation = self.reset()
            ce_total = 0.0
            total_reward = 0.0

            for step in range(MAX_STEPS):
                state_idx = self.agent.state.analog2digitize(observation)
                action = np.argmax(self.agent.state.q_table[state_idx, :])

                observation_next, reward_step, done, ce_step = self.step(action)
                ce_total += ce_step
                total_reward += reward_step
                observation = observation_next

                if done:
                    break

            reward_final = self.compute_release_reward(self.z, ce_total)
            total_reward += reward_final
            returns.append(total_reward)

        return np.mean(returns)'''

# =========================
# メイン
# =========================
def main():
    env = Environment()
    env.run()

    # ① エピソードごとの報酬
    plt.figure()
    plt.plot(env.reward_history)
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("Episode reward")
    plt.show()

    # ② greedy 評価の推移
    '''
    plt.figure()
    plt.plot(env.eval_history)
    plt.xlabel("Episode")
    plt.ylabel("Greedy evaluation return")
    plt.title("Greedy policy performance")
    plt.show()
    '''

    # ③ 移動平均（おまけ）
    window = 100
    rewards = np.array(env.reward_history)
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.figure()
        plt.plot(np.arange(window, window + len(moving_avg)), moving_avg)
        plt.xlabel("Episode")
        plt.ylabel("Average reward (last 100 episodes)")
        plt.title("2-link throwing Q-learning performance")
        plt.show()

    np.set_printoptions(threshold=np.inf)
    print(env.agent.state.q_table)

if __name__ == "__main__":
    main()
