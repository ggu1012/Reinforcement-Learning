import numpy as np
from visualize_train import draw_value_image, draw_policy_image

# left, right, up, down
ACTIONS = [np.array([0, -1]),
           np.array([0, 1]),
           np.array([-1, 0]),
           np.array([1, 0])]

TRAINING_EPISODE_NUM = 800000

class AGENT:
    def __init__(self, env, is_upload=False):

        self.ACTIONS = ACTIONS
        self.env = env
        HEIGHT, WIDTH = env.size()
        self.state = [0,0]

        if is_upload:   # Test
            mcc_results = np.load('./result/mcc.npz')
            self.V_values = mcc_results['V']
            self.Q_values = mcc_results['Q']
            self.policy = mcc_results['PI']
        else:          # For training
            self.V_values = np.zeros((HEIGHT, WIDTH))
            self.Q_values = np.zeros((HEIGHT, WIDTH, len(self.ACTIONS)))
            self.policy = np.zeros((HEIGHT, WIDTH,len(self.ACTIONS)))+1./len(self.ACTIONS)



    def initialize_episode(self):
        HEIGHT, WIDTH = self.env.size()
        while True:
            i = np.random.randint(HEIGHT)
            j = np.random.randint(WIDTH)
            state = [i, j]
            if (state in self.env.goal) or (state in self.env.obstacles):
                continue
            break
            # if (state not in self.env.goal) and (state not in self.env.obstacles):
            #     break
        return state



    def Monte_Carlo_Control(self, discount=1.0, alpha=0.01, max_seq_len=500,
                            epsilon=0.3, decay_period=20000, decay_rate=0.9):

        for episode in range(TRAINING_EPISODE_NUM):
            state = self.initialize_episode()

            done = False
            timeout = False
            seq_len = 0
            history = []

            # Sequence generation
            history.append((state, action, next_state, reward))

            # Q Value and policy update
            self.Q_values[i][j][a] += alpha * (cum_reward - self.Q_values[i][j][a])


        self.V_values = np.max(self.Q_values, axis=2)
        draw_value_image(1, np.round(self.V_values, decimals=2), env=self.env)
        draw_policy_image(1, np.round(self.policy, decimals=2), env=self.env)
        np.savez('./result/mcc.npz', Q=self.Q_values, V=self.V_values, PI=self.policy)
        return self.Q_values, self.V_values, self.policy



    def get_action(self, state):
        i,j = state
        return np.random.choice(len(ACTIONS), 1, p=self.policy[i,j,:].tolist()).item()


    def get_state(self):
        return self.state

