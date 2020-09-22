import os
import numpy as np
import torch

# ----------------------------------------------------------------------------------
#                           REPLAY BUFFER
# Code based on:
# https://github.com/sfujim/TD3/blob/master/utils.py
#
# ----------------------------------------------------------------------------------

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e4)): # changed max size default
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def add(self, state, action, next_state, reward, done, state_shape):
        self.state[self.ptr] = state['image'].reshape(state_shape)
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state['image'].reshape(state_shape)
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )


# ----------------------------------------------------------------------------------
#                               LOGGING
# ----------------------------------------------------------------------------------

def save_session(model, args, episode):
    # appending Episode number to the file, also see the overall end timestep, so I
    # know how much the agent has seen
    # ------------------------------------------------------------------------------
    path = os.path.join(args.save_dir, str(episode))
    if not os.path.exists(path):
        os.makedirs(path)

    # Save the model and optimizer state
    # ------------------------------------------------------------------------------
    torch.save(model.state_dict(), os.path.join(path, 'model.pth'))
    print('Successfully saved model')

    #save to Comet Asset Tab
    if args.comet:
        args.experiment.log_asset(file_data= args.save_dir+'/'+str(episode)+'/' +'model.pth', file_name= 'ep'+str(episode)+'model.pth' )


def load_session(model, optim, args):
    # Bring the model back, and restart. (With exception handling)
    # ------------------------------------------------------------------------------
    try:
        start_epoch = int(args.load_dir.split('/')[-1])
        model.load_state_dict(torch.load(os.path.join(args.load_dir, 'model.pth')))
        print('Successfully loaded model')
    except Exception as e:
        ipdb.set_trace()
        print('Could not restore session properly')

    return model, optim, start_epoch


# From https://github.com/lcswillems/rl-starter-files

import json
import re
import gym



def get_obss_preprocessor(obs_space):
    # Check if obs_space is an image space
    if isinstance(obs_space, gym.spaces.Box):
        obs_space = {"image": obs_space.shape}


    # Check if it is a MiniGrid observation space
    elif isinstance(obs_space, gym.spaces.Dict) and list(obs_space.spaces.keys()) == ["image"]:
        obs_space = {"image": obs_space.spaces["image"].shape, "text": 100}
        vocab = Vocabulary(obs_space["text"])


    else:
        raise ValueError("Unknown observation space: " + str(obs_space))

    return obs_space


class Vocabulary:
    """A mapping from tokens to ids with a capacity of `max_size` words.
    It can be saved in a `vocab.json` file."""

    def __init__(self, max_size):
        self.max_size = max_size
        self.vocab = {}

    def load_vocab(self, vocab):
        self.vocab = vocab

    def __getitem__(self, token):
        if not token in self.vocab.keys():
            if len(self.vocab) >= self.max_size:
                raise ValueError("Maximum vocabulary capacity reached")
            self.vocab[token] = len(self.vocab) + 1
        return self.vocab[token]
