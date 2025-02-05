import numpy as np
import itertools
import gym
import random
from gym import spaces
from gym.utils import seeding
import torch
import torch.nn.functional as F
import torch.nn as nn
import wandb

device = 'cpu'

class MhPolicy(nn.Module):
    def __init__(self, input_dim, output_dim, rank, custom_lr=0.01, omega=0.5, momentum=0.0):
        super(MhPolicy, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.rank = rank
        self.omega = omega

        self.H = nn.Parameter(torch.randn(input_dim, rank))
        self.V = nn.Parameter(torch.randn(output_dim, rank))

        self.normal_init()

        self.vK = 0
        self.vVp = 0

        self.custom_lr = custom_lr
        self.momentum = momentum

    def normal_init(self):
        with torch.no_grad():
            matrix = torch.empty(self.input_dim, self.output_dim).to(device)
            torch.nn.init.normal_(matrix, mean=0.0, std=0.05)
            U, S, V = torch.linalg.svd(matrix,full_matrices=False)
            self.H.data = U[:,:self.rank]*(S[:self.rank].view(1,-1))
            self.H.data = torch.ones(self.input_dim, self.rank)
            self.H.data = self.project_sphere(self.H)
            self.V.data = V[:self.rank,:].T

    def generate_prob(self, state_inx):
        x = self.H[state_inx,:]@self.V.T
        return x*x
    
    def project_tangent(self,point,vector):
        return vector - torch.sum(vector*point,1).unsqueeze(1)*point

    def project_sphere(self,matrix):
        return F.normalize(matrix, p=2, dim=1)

    def project_Vp(self,matrix):
        return matrix - self.V@(self.V.T@matrix)

    def factorSH(self):
        temp = 2*self.omega*torch.eye(self.rank).to(device) + (self.H.T@self.H)
        return temp

    def apply_custom_update(self):
        temp_Vp = self.V.grad@torch.inverse(self.factorSH())
    
        # momentum
        self.vK = self.project_tangent(self.H,self.momentum*self.vK+self.H.grad)
        self.vVp = self.project_Vp(self.momentum*self.vVp+temp_Vp) 

        # update H and V via (1st-) Rie retraction 
        t = self.custom_lr
        Z = self.V - t * self.vVp
        ZTZ = Z.T @ Z
        eigvals, eigvecs = torch.linalg.eigh(ZTZ)
        sqrt_ZTZ_inv = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals)) @ eigvecs.T
        self.V.data = Z @ sqrt_ZTZ_inv
        self.H.data = self.project_sphere(self.H-t*self.vK)

        self.zero_grad()

    def reorth_V(self):
        UV, SV, VV = torch.linalg.svd(self.V.data , full_matrices=False)
        self.V.data = UV@VV

class MhRL:
    def __init__(self,
                 env,
                 state_map,
                 action_map,
                 state_reverse_map,
                 action_reverse_map,
                 n_states,
                 n_actions,
                 step_state,
                 step_action,
                 decimal_state,
                 decimal_action,
                 episodes=100000,
                 max_steps=1000,
                 alpha=0.01,
                 gamma=0.99,
                 r=5,
                 args=None):
        """
        :param env: gym.envs
            OpenAI Gym environment.
        :param state_map: dict
            Dictionary mapping state indices to codified states.
        :param action_map: dict
            Dictionary mapping action indices to codified actions.
        :param state_reverse_map: dict
            Dictionary mapping codified states to state indices.
        :param action_reverse_map: dict
            Dictionary mapping codified actions to action indices.
        :param n_states: int
            Number of states.
        :param n_actions: int
            Number of actions.
        :param step_state: float
            Step size for state discretization.
        :param step_action: float
            Step size for action discretization.
        :param decimal_state: int
            Number of decimals for state discretization.
        :param decimal_action: int
            Number of decimals for action discretization.
        :param episodes: int
            Number of episodes.
        :param max_steps: int
            Maximum steps per episode.
        :param alpha: float
            Learning rate.
        :param gamma: float
            Discount factor.
        :param r: int
            Dimension of the latent space.
        """

        self.env = env
        self.state_map = state_map
        self.action_map = action_map
        self.state_reverse_map = state_reverse_map
        self.action_reverse_map = action_reverse_map
        self.n_states = n_states
        self.n_actions = n_actions
        self.step_state = step_state
        self.step_action = step_action
        self.decimal_state = decimal_state
        self.decimal_action = decimal_action
        self.episodes = episodes
        self.max_steps = max_steps
        self.alpha = alpha
        self.gamma = gamma
        self.r = r
        self.args = args

        # Policy parameterized by a matrix
        self.policy = MhPolicy(self.n_states,self.n_actions,rank=self.r, custom_lr=self.alpha)

        # Record rewards and steps
        self.cumulative_rewards = []
        self.steps = []


    def get_s_idx(self, st):
        """
        :param st: np.array
            State to obtain the row index of the policy matrix.
        :return: int
            Row index of the policy matrix.
        """
        st_ = np.array([np.arctan(st[1] / st[0]), st[2]])
        st_ = np.array([self.step_state * (np.round(s / self.step_state)) for s in st_])
        return self.state_reverse_map[str(np.around(st_, self.decimal_state) + 0.)]

    def get_a_idx(self, at):
        """
        :param at: np.array
            Action to obtain the column index of the policy matrix.
        :return: int
            Column index of the policy matrix.
        """
        at_ = [self.step_action * (np.round(a / self.step_action)) for a in at]
        return self.action_reverse_map[str(np.around(at_, self.decimal_action) + 0.)]

    def choose_action(self, state_idx):
        """Choose an action based on the current policy."""
        probs = self.policy.generate_prob(state_idx)
        action_idx = np.random.choice(np.arange(self.n_actions), p=probs.detach().numpy())
        return self.action_map[action_idx], action_idx, probs

    def train(self):
        timestep = 0
        for episode in range(self.episodes):
            state = self.env.reset(upright=True)
            state_idx = self.get_s_idx(state)

            episode_rewards = []
            episode_actions = []
            episode_states = []
            episode_logprob = []

            for step in range(self.max_steps):
                action, action_idx, probs = self.choose_action(state_idx)
                next_state, reward, done, _ = self.env.step(action)

                theta = np.arctan(next_state[1] / next_state[0])
                done = True if ((theta > np.pi / 4) | (theta < -np.pi / 4)) else False

                episode_states.append(state_idx)
                episode_actions.append(action_idx)
                episode_rewards.append(reward)
                episode_logprob.append(torch.log(probs[action_idx]))

                state = next_state
                state_idx = self.get_s_idx(next_state)

                timestep += 1
                if done:
                    break

            self.update_policy(episode_states, episode_actions, episode_rewards,episode_logprob)

            if self.args:
                if (episode) % self.args.test_fre == 0:
                    r, steps = self.test(self.max_steps)
                    print(f"Episode {episode}: Reward {r}, Length {steps}")
                    
                    # wandb log
                    if self.args.track:
                        wandb.log({"episode": episode, "reward": r, "length": steps, "timestep": timestep})

            else:
                if (episode % 10) == 0:
                    r, steps = self.test(self.max_steps)
                    print(f"Episode {episode}: Reward {r}, Length {steps}")


    def update_policy(self, states, actions, rewards,logprob):
        """Update the policy using REINFORCE."""
        returns = self.compute_returns(rewards)
        loss = 0

        for t in range(len(states)):
            state_idx = states[t]
            action_idx = actions[t]
            G_t = returns[t]

            # Compute policy gradient
            loss = -G_t*logprob[t]

            # Update policy
            loss.backward()
            with torch.no_grad():
                self.policy.apply_custom_update()


    def compute_returns(self, rewards):
        """Compute the discounted returns for an episode."""
        returns = np.zeros(len(rewards))
        G = 0
        for t in reversed(range(len(rewards))):
            G = rewards[t] + self.gamma * G
            returns[t] = G
        return returns

    def test(self, n_steps):
        """Run a test episode using the trained policy."""
        """
        :param n_steps: int
            Number of steps of the greedy episode.
        :return: tuple
            Cumulative reward of the episode and number of steps.
        """

        s = self.env.reset(upright=True)
        cum_r = 0

        for i in range(n_steps):
            state_idx = self.get_s_idx(s)
            action, _, _ = self.choose_action(state_idx)
            next_state, r, _, _ = self.env.step(action)
            
            theta = np.arctan(next_state[1] / next_state[0])
            done = (theta > 1.0) | (theta < -1.0)

            cum_r += r
            s = next_state

            if done:
                break

        return cum_r, i


class Mapper:
    def __init__(self):

        self.min_theta = -1.0 # 50 grados
        self.max_theta = 1.0 # 50 grados
        self.min_theta_dot = -5.0
        self.max_theta_dot = 5.0

        self.min_joint_effort = -2.0
        self.max_joint_effort = 2.0

    def get_map(self, iterable):
        """
        :param iterable: list
            Elements of the state space.
        :return: dict
            Cartesian product of the state space.
        """

        mapping = [np.array(combination) for combination in itertools.product(*iterable)]
        reverse_mapping = {str(mapping[i]):i for i in range(len(mapping))}

        return mapping, reverse_mapping

    def get_state_map(self, step, decimal):
        """
        :param step: float
            Discretization step.
        :param decimal: int
            Precision.
        :return: dict
            Map of states and indices.
        """

        theta = np.around(np.arange(self.min_theta, self.max_theta + step, step), decimal) + 0.
        theta_dot = np.around(np.arange(self.min_theta_dot, self.max_theta_dot + step, step), decimal) + 0.

        return self.get_map([theta, theta_dot])

    def get_action_map(self, step, decimal):
        """
        :param step: float
            Discretization step.
        :param decimal: int
            Precision.
        :return: dict
            Map of actions and indices.
        """

        joint_effort = np.around(np.arange(self.min_joint_effort, self.max_joint_effort + step, step), decimal) + 0.

        return self.get_map([joint_effort])


class MapperReshape:
    def __init__(self):

        self.min_theta = -1.0 # 50 grados
        self.max_theta = 1.0 # 50 grados
        self.min_theta_dot = -5.0
        self.max_theta_dot = 5.0

        self.min_joint_effort = -2.0
        self.max_joint_effort = 2.0

        self.step = 0.1
        self.decimal = 1

        state_key = []
        action_key = []

        theta = np.around(np.arange(self.min_theta, self.max_theta + self.step, self.step), self.decimal) + 0.
        theta_dot = np.around(np.arange(self.min_theta_dot, self.max_theta_dot + self.step, self.step), self.decimal) + 0.

        joint_effort = np.around(np.arange(self.min_joint_effort, self.max_joint_effort + self.step, self.step), self.decimal) + 0.

        grid = [theta, theta_dot, joint_effort]

        self.forward_map = [combination for combination in itertools.product(*grid)]
        random.shuffle(self.forward_map)

        self.n_rows = int(np.ceil(np.sqrt(len(self.forward_map))))
        self.n_cols = self.n_rows
        matrix_idx = []
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                matrix_idx.append((i, j))

        self.reverse_map = dict(zip(self.forward_map, matrix_idx))

    def update_state_key(self, state):

        s_theta = np.arctan(state[1] / state[0])
        s_theta_dot = state[2]

        theta = np.around(self.min_theta + self.step * ((s_theta + self.max_theta) // self.step), self.decimal) + 0.
        theta_dot = np.around(self.min_theta_dot + self.step * ((s_theta_dot + self.max_theta_dot) // self.step), self.decimal) + 0.

        theta = np.clip(theta, self.min_theta, self.max_theta)
        theta_dot = np.clip(theta_dot, self.min_theta_dot, self.max_theta_dot)

        self.state_key = [theta, theta_dot]

    def update_action_key(self, action):

        act = np.around(self.min_joint_effort + self.step * ((action[0] + self.max_joint_effort) // self.step), self.decimal) + 0.
        act = np.clip(act, self.min_joint_effort, self.max_joint_effort)

        self.action_key = [act]

    def get_matrix_index(self):
        key = tuple(self.state_key + self.action_key)
        return self.reverse_map[key]





class PendulumEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, g=10.0):
        self.max_speed = 8
        self.max_torque = 2.
        self.dt = .05
        self.g = g
        self.m = 1.
        self.l = 1.
        self.viewer = None

        high = np.array([1., 1., self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque, shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        costs = self.angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)

        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = np.array([newth, newthdot])
        return self._get_obs(), -costs, False, {}

    def reset(self, upright=False):
        high = np.array([np.pi, 1])
        if upright:
            self.state = [np.random.rand()/100, np.random.rand()/100]
        else:
            self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = "resources/clockwise.png"
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


    def angle_normalize(self, x):
        return ((x+np.pi) % (2*np.pi)) - np.pi

