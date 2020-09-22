from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils


# Used for Atari
class Conv_Q(nn.Module):
    def __init__(self, frames, num_actions):
        super(Conv_Q, self).__init__()
        self.c1 = nn.Conv2d(frames, 32, kernel_size=8, stride=4)
        self.c2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.c3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.l1 = nn.Linear(3136, 512)
        self.l2 = nn.Linear(512, num_actions)


    def forward(self, state):
        q = F.relu(self.c1(state))
        q = F.relu(self.c2(q))
        q = F.relu(self.c3(q))
        q = F.relu(self.l1(q.reshape(-1, 3136)))
        return self.l2(q)


# Used for Box2D / Toy problems
class FC_Q(nn.Module):
    def __init__(self, state_dim, num_actions):
        super(FC_Q, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, num_actions)


    def forward(self, state):
        q = F.relu(self.l1(state))
        q = F.relu(self.l2(q))
        return self.l3(q)


class DQN(object):
    def __init__(self, parameters, env_properties, device):
        self.device = device

        print("---------------------------------------")
        print("---------------  DQN   ----------------")
        print("---------------------------------------")


        # Make Q network, target network and initialize optimizer
        self.Q = Conv_Q(4, env_properties["num_actions"]).to(self.device) if env_properties["atari"] \
            else FC_Q(env_properties["state_dim"], env_properties["num_actions"]).to(self.device)
        self.Q_target = deepcopy(self.Q)
        self.Q_optimizer = getattr(torch.optim, parameters["optimizer"])(
          self.Q.parameters(), **parameters["optimizer_parameters"]
        )

        # Parameters for train()
        self.discount = parameters["discount"]

        # Select target update rule
        # copy: copy full target network every "target_update_frequency" iterations
        # polyak: update every timestep with proportion tau
        self.maybe_update_target = self.polyak_target_update if parameters["polyak_target_update"] \
            else self.copy_target_update
        self.target_update_frequency = parameters["target_update_frequency"] \
            / parameters["update_frequency"]
        self.tau = parameters["tau"]

        # Parameters for exploration + Compute linear decay for epsilon
        self.initial_epsilon = parameters["initial_epsilon"]
        self.end_epsilon = parameters["end_epsilon"]
        self.slope = (self.end_epsilon - self.initial_epsilon) \
            / parameters["epsilon_decay_period"] * parameters["update_frequency"]

        # Parameters for evaluation
        self.state_shape = (-1, 4, 84, 84) if env_properties["atari"] else (-1, env_properties["state_dim"]) # need to unroll gridworld into the right form
        self.evaluation_epsilon = parameters["evaluation_epsilon"]
        self.num_actions = env_properties["num_actions"]

        # Number of training iterations
        self.iterations = 0

        # List of elements returned by train(), to be displayed by logger
        self.display_list = ["Batch_Q", "Batch_TD", "Epsilon"]


    def select_action(self, state, eval=False):
        eps = self.evaluation_epsilon if eval \
            else max(self.slope * self.iterations + self.initial_epsilon, self.end_epsilon)

        # Select action according to policy with probability (1-eps)
        # otherwise, select random action
        if np.random.uniform(0,1) > eps:
            with torch.no_grad():
                #state = utils.get_obss_preprocessor(state)
                state = torch.FloatTensor(state.sum()['image']).reshape(self.state_shape).to(self.device)
                return int(self.Q(state).argmax(1))
        else:
            return np.random.randint(self.num_actions)
            # return 0


    def train(self, replay_buffer, batch_size):
        # Sample replay buffer
        state, action, next_state, reward, done = replay_buffer.sample(batch_size)

        # Compute the target Q value
        with torch.no_grad():
            target_Q = (
                reward + done * self.discount *
                self.Q_target(next_state).max(1, keepdim=True)[0]
            )

        # Get current Q estimate
        current_Q = self.Q(state).gather(1, action.long())

        # Compute critic loss
        critic_loss = F.smooth_l1_loss(current_Q, target_Q)

        # Optimize the critic
        self.Q_optimizer.zero_grad()
        critic_loss.backward()
        self.Q_optimizer.step()

        # Update target network by polyak or full copy every X iterations.
        self.iterations += 1
        self.maybe_update_target()

        # Return values for logger to display
        eps = max(self.slope * self.iterations + self.initial_epsilon, self.end_epsilon)
        return {
            "Batch_Q": current_Q.cpu().data.numpy().mean(),
            "Batch_TD": critic_loss.cpu().data.numpy(),
            "Epsilon": eps
        }


    def polyak_target_update(self):
        for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
           target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def copy_target_update(self):
        if self.iterations % self.target_update_frequency == 0:
             self.Q_target.load_state_dict(self.Q.state_dict())


    def save(self, filename):
        torch.save(self.Q.state_dict(), filename + "/model.pth")


    def load(self, filename):
        self.Q.load_state_dict(torch.load(filename + "/model.pth"))
