from comet_ml import Experiment

import numpy as np
import torch

import argparse
import os
import json
import ipdb
from tqdm import tqdm

import utils
import DQN

import gym
import gym_minigrid
import neuroRL

import time


# ----------------------------------------------------------------------------------
#                             RL GYM EXPERIMENTS!
# ----------------------------------------------------------------------------------

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
    test_env_name = env_name.replace('-Random','')
    eval_env = gym.make(test_env_name)
    # import ipdb
    # ipdb.set_trace()
    eval_env.seed(seed)

    avg_reward = 0.
    AvgNscore = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        ep_count = 0
        xgoal, ygoal = 8.,8.
        x,y = eval_env.agent_pos
        while not done:
            ep_count += 1
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
        AvgNscore += ( (ygoal- y) + (xgoal - x) ) / ep_count

    avg_reward /= eval_episodes
    AvgNscore /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f},{AvgNscore:.3f}")
    print("---------------------------------------")
    return avg_reward, AvgNscore

if __name__ == "__main__":

    # Where the command line magic happens.
    # ------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()

    parser.add_argument("--comet", action='store_true', default=False, help='to use https://www.comet.ml/joshholla/neuro-rl/view/new for logging')
    parser.add_argument("--namestr",type=str,default='DQN',help='additional info in output filename to describe experiments')
    parser.add_argument("--tag",type=str,default='v0.1.0',help='additional info in output filename to describe experiments')

    parser.add_argument("--env_name", default="MiniGrid-Empty-Random-10x10-v0")  # Custom gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds

    parser.add_argument("--algorithm", default="DQN") # C51, DQN, QRDQN
    parser.add_argument("--debug",action='store_true',default=False,help='to prevent logging even to disk, when debugging.')

    parser.add_argument("--load_dir",type=str,default=None,help='use existing model. Load model from _ directory')
    parser.add_argument("--save_dir",type=str,default='./results/',help='directory for saving session')
    parser.add_argument("--max_timesteps", default=1e6, type=int)
    parser.add_argument("--start_timesteps", default=25e3, type=int)
    parser.add_argument("--evaluate_every",type=int, default=5e3, help='evaluate every _ timesteps' )
    parser.add_argument("--save_every",type=int, default=5e4, help='save every _ timesteps' )
    #parser.add_argument("--log_every",type=int, default=, help='log every _ timesteps' )

    parser.add_argument("--save_models", default=True)  # Whether or not models are saved

    parser.add_argument("--optimizer",type=str, default='Adam', help='optimization algorithm to use')
    parser.add_argument("--discount", default=0.99, type=float)# buy one get one
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--polyak_target_update",action='store_true',default=False,help='Are we using polyak target updates? yay or nay')
    parser.add_argument("--target_update_frequency", default=200, type=int)
    parser.add_argument("--update_frequency", default=1, type=int)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--initial_epsilon", default = 0.1 , type = int)
    parser.add_argument("--end_epsilon", default = 0.1  , type = int)
    parser.add_argument("--epsilon_decay_period", default = 5e2 , type = int)
    parser.add_argument("--evaluation_epsilon", default = 1e-6 , type = int)

    parser.add_argument("--atari",action='store_true',default=False)
    parser.add_argument("--batch_size", default=256, type=int) # for replay buffer.
    parser.add_argument("--normalize_score", action='store_true', default=False)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Configure Logging.
    # a settings.json file should be included for logging to comet. (not pushed to git)
    # ------------------------------------------------------------------------------

    if args.debug:
        args.comet = False # We don't need no logging when debugging.


    if args.comet:
        if args.comet==True and os.path.isfile("./rlscripts/settings.json"):
            with open('./rlscripts/settings.json') as f:
                data = json.load(f)
            args.comet_apikey = data["api_key"]
            args.comet_username = data["workspace"]
            args.comet_project = data["project_name"]
        experiment = Experiment(api_key=args.comet_apikey, project_name=args.comet_project, workspace=args.comet_username, auto_output_logging="native")
        experiment.set_name(args.namestr)
        experiment.add_tag(args.tag)
        args.experiment = experiment
        #experiment.log_parameter(args) # logs args TODO



    # Because we all like reproducibility (¯\_(ツ)_/¯)
    # ------------------------------------------------------------------------------
    seed = 42 # np.random.randint(100)  # if we want to mix up seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    env = gym.make(args.env_name)

    env.seed(seed)
    #state_dim = env.observation_space.shape[0]
    obs_space = utils.get_obss_preprocessor(env.observation_space)
    #state_dim = obs_space.shape['image'] # we want a number. [NxM tiles, then
    # obj, color, type]
    total = 1
    for idx, value in enumerate(obs_space['image']):
        total *= value
    state_dim = total
    num_actions = env.action_space.n

    if args.comet:
        print("---------------------------------------")
        print("Comet Experiment: %s" % (args.namestr))
        print("Seed : %s" % (seed))
        print("---------------------------------------")

    args.optimizer_parameters ={}
    args.optimizer_parameters.update(lr=args.lr)

    args.parameters = {}
    args.parameters.update(discount=args.discount)
    args.parameters.update(optimizer=args.optimizer)
    # add optimizer_parameters
    args.parameters.update(optimizer_parameters=args.optimizer_parameters)
    args.parameters.update(polyak_target_update=args.polyak_target_update)
    args.parameters.update(target_update_frequency=args.target_update_frequency)
    args.parameters.update(update_frequency=args.update_frequency)
    args.parameters.update(tau=args.tau)
    args.parameters.update(initial_epsilon=args.initial_epsilon)
    args.parameters.update(end_epsilon=args.end_epsilon)
    args.parameters.update(epsilon_decay_period=args.epsilon_decay_period)
    args.parameters.update(evaluation_epsilon=args.evaluation_epsilon)

    args.env_properties = {}
    args.env_properties.update(num_actions=num_actions)
    args.env_properties.update(atari=args.atari)
    args.env_properties.update(state_dim=state_dim)

    if (args.algorithm=="DQN"):
        model = DQN.DQN(args.parameters, args.env_properties, device)
    else:
        model  = DQN.DQN(args.parameters, args.env_properties, device)
    replay_buffer = utils.ReplayBuffer(state_dim, 1)

    # Evaluate untrained policy
    evaluations = [eval_policy(model, args.env_name, args.seed)]

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    action_list = []

    # Starting at the same orientation. 0
    # (y_goal - y) + (x_goal - x) + 1 to turn for the manhattan distance.
    y_goal = 8 # TODO - make this adapt to the env.
    x_goal = 8
    x,y = env.agent_pos
    state_action_table = {}

    for t in range(int(args.max_timesteps)):

        episode_timesteps += 1
        start_time = time.time()

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                model.select_action(np.array(state))
            )

        # add action taken for logging.
        action_list.append(action)

        # I'm going to update a table with states and actions as well:
        state_action = ''.join(str(e) for e in env.agent_pos)
        state_action+=(str(env.agent_dir))
        state_action+=(str(action))
        # state_action = str(env.agent_pos).join(str(env.agent_dir)).join(str(action))   # append position (state) and action for Q value
        if state_action in state_action_table:
            state_action_table[state_action] += 1
        else:
            state_action_table[state_action] = 1

        # Perform action
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_timesteps < env.max_steps else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool, state_dim)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            model.train(replay_buffer, args.batch_size)

        """
        + Things being log:
        + Action Sequences
        + Time (wall clock)
        + Rewards
        + Q values
        + Visitation counts
        + Normalized Scores
        Pending to log:
        + Q value distributions for distributional methods
        """
        if done:
            end_time = time.time()
            episode_time = end_time - start_time
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.2f}")
            if args.comet:
                args.experiment.log_metric("Training Episode Reward", episode_reward, step=t)

                # Different step here
                utils.save_session(model.Q, args, episode_num)
                args.experiment.log_metric("Episode Timestep", episode_timesteps, step = episode_num)
                args.experiment.log_metric("Training Episode Reward", episode_reward, step=episode_num)
                args.experiment.log_metric("Wall Time", time.time(), step=episode_num)
                args.experiment.log_metric("Episode Length", episode_time, step= episode_num)

                # Debug: Changed to log_parameter
                args.experiment.log_parameter("Action Sequence", str(action_list), step = episode_num)
                print(f"State Visitation Table {state_action_table} Episode Num: {episode_num+1}")

                # Log the normalized score:
                if args.normalize_score:
                    # Make it go from 0 to 1.
                    Nscore = ( (y_goal - y) + (x_goal - x) ) / episode_timesteps
                    args.experiment.log_metric("Normalized Score Train", Nscore, step = episode_num)

            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            action_list = []

        # Evaluate episode - Evaluate each episode.
        # ----------------------------------------------------------------------
        evaluations.append(eval_policy(model, args.env_name, args.seed))
        # [0]
        if args.comet:
            args.experiment.log_metric("Eval Reward", evaluations[-1][0], step=t)
            args.experiment.log_metric("Normalized Score Eval",evaluations[-1][1], step = t)

    args.experiment.log_others(state_action_table)
    args.experiment.end()

    # ------------------------------------------------------------------------------
    #                                         FIN
    # ------------------------------------------------------------------------------
