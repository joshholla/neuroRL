#!/usr/bin/env python3

from comet_ml import Experiment

import time
import argparse
import os
import json
import numpy as np
import gym
import gym_minigrid
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window
from utils import load_struct, save_struct

def redraw(img):
    if args.expert_view:
        img = env.render('rgb_array', tile_size=args.tile_size)

    window.show_img(img)

def reset():
    global action_sequence
    global state_draw_sequence
    global state_position_sequence
    global decision_time_logs
    global reward_logs
    global episode_num
    global save_name
    global key_press_sequence

    if args.seed != -1:
        env.seed(args.seed)

    obs = env.reset()
    episode_num += 1

    # Save to disk

    logs = {
        "action_sequence": action_sequence,
        "state_sequence": state_position_sequence,
#        "grid_viz_sequence": state_draw_sequence,
        "key_press_sequence": key_press_sequence,
        "decision_times": decision_time_logs,
        "rewards": reward_logs
    }

    path  = os.path.join(args.save_dir, save_name)
    if not os.path.exists(path):
        os.makedirs(path)

    working_name = save_name
    working_name += '_'
    working_name += str(episode_num - 1)

    if (episode_num - 1 != 0):
        file_path = os.path.join(path, working_name+'.pkl')
        save_struct(logs, file_path)

        if args.comet:
            args.experiment.log_asset(file_data= file_path, file_name = working_name)

    if hasattr(env, 'mission'):
        print('Mission: %s' % env.mission)
        window.set_caption(env.mission)

    redraw(obs)

    # Resetting the logging stuff.
    action_sequence = []
#    state_draw_sequence = []
    state_position_sequence = []
    decision_time_logs = []
    reward_logs = []
    key_press_sequence = []


def step(action, decision_time):
    obs, reward, done, info = env.step(action)

    global state_position_sequence
    global state_draw_sequence
    global decision_time_logs
    global reward_logs
    global state_action_table
    state_draw_sequence.append(env.__str__)
    state_position_sequence.append(env.agent_pos) # check dim of this return
    decision_time_logs.append(decision_time)
    reward_logs.append(reward)

    info = 'episode=%s, step=%s, reward=%.2f, decision time=%.2f, time=%.2f' % (episode_num, env.step_count, reward, decision_time, time.time())
    print('episode=%s, step=%s, reward=%.2f, decision time=%.2f, time=%.2f' % (episode_num, env.step_count, reward, decision_time, time.time()))
    print(f"State Visitation Table {state_action_table} Episode Num: {episode_num+1}")
    if args.comet:
        args.experiment.log_metric("reward", reward, step= env.step_count)
        args.experiment.log_others(state_action_table)
        args.experiment.log_text(info,step=episode_num)
    if done:
        print('done!')
        reset()
    else:
        redraw(obs)

def key_handler(event):
    print('pressed', event.key)

    if env.step_count == 0:
        state_position_sequence.append(env.agent_pos)


    global last_time
    global key_press_sequence
    current_time = time.time()
    decision_time = current_time - last_time
    last_time = current_time
    if args.comet:
        args.experiment.log_parameter("Pressed", event.key, step=env.step_count)
        args.experiment.log_metric("Decision Time", decision_time, step = env.step_count)

    key_press_sequence.append(event.key)
    # ['left', 'right', 'up'] Getting mapped to the following keyboard inputs:
    key_choices= ['1', '3', '2']

    global action_sequence
    # I'm going to update a table with states and actions as well:
    state_action = ''.join(str(e) for e in env.agent_pos)
    state_action+=(str(env.agent_dir))
    #state_action+=(str(action))

    if event.key == 'escape':
        if args.comet:
            args.experiment.log_parameter("Action", "escape" ,step=env.step_count)
        action_sequence.append('escape')
        window.close()
        return

    if event.key == 'backspace':
        if args.comet:
            args.experiment.log_parameter("Action", "backspace" ,step=env.step_count)
        action_sequence.append('backspace')
        reset()
        return

    if event.key == key_choices[order_choices[0]]: #'left':
        if args.comet:
            args.experiment.log_parameter("Action", "left" ,step=env.step_count)
        action_sequence.append('left')
        state_action+='1'
        update_visitation(state_action)
        step(env.actions.left, decision_time)
        return
    if event.key == key_choices[order_choices[1]]: # 'right':
        if args.comet:
            args.experiment.log_parameter("Action", "right" ,step=env.step_count)
        action_sequence.append('right')
        state_action+='3'
        update_visitation(state_action)
        step(env.actions.right, decision_time)
        return
    if event.key == key_choices[order_choices[2]]: # 'up':
        if args.comet:
            args.experiment.log_parameter("Action", "up" ,step=env.step_count)
        action_sequence.append('up')
        state_action+='2'
        update_visitation(state_action)
        step(env.actions.forward, decision_time)
        return

    # Spacebar
    if event.key == ' ':
        if args.comet:
            args.experiment.log_parameter("Action", "spacebar" ,step=env.step_count)
        action_sequence.append('spacebar')
        step(env.actions.toggle, decision_time)
        return
    if event.key == 'pageup':
        if args.comet:
            args.experiment.log_parameter("Action", "pick up" ,step=env.step_count)
        action_sequence.append('pick up')
        state_action+='4'
        update_visitation(state_action)
        step(env.actions.pickup, decision_time)
        return
    if event.key == 'pagedown':
        if args.comet:
            args.experiment.log_parameter("Action", "drop" ,step=env.step_count)
        action_sequence.append('drop')
        state_action+='5'
        update_visitation(state_action)
        step(env.actions.drop, decision_time)
        return

    if event.key == 'enter':
        if args.comet:
            args.experiment.log_metric("Action", "done" , step=env.step_count)
        action_sequence.append('enter')
        state_action+='6'
        update_visitation(state_action)
        step(env.actions.done, decision_time)
        return

def update_visitation(state_action):
    global state_action_table
    if state_action in state_action_table:
        state_action_table[state_action] += 1
    else:
        state_action_table[state_action] = 1


parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    help="gym environment to load",
    default='MiniGrid-Empty-Random-10x10-v0'
)
parser.add_argument(
    "--seed",
    type=int,
    help="random seed to generate the environment with",
    default=-1
)
parser.add_argument(
    "--save_dir",
    type=str,
    help="location of saved log files",
    default="logs/"
)
parser.add_argument(
    "--tile_size",
    type=int,
    help="size at which to render tiles",
    default=32
)
parser.add_argument(
    '--agent_view',
    default=True,
    help="draw the agent sees (partially observable view)",
    action='store_false'
)
parser.add_argument(
    '--debug',
    action='store_true',
    default=False,
    help="Prevent logging when testing"
)
parser.add_argument(
    '--comet',
    action='store_true',
    default=False,
    help='to use https://www.comet.ml/joshholla/neuro-rl/view/new for logging'
)
parser.add_argument(
    '--namestr',
    type=str,
    default='neuroRLHumanRun',
    help='additional info to describe experiments'
)
parser.add_argument(
    '--tag',
    type=str,
    default='Human',
    help='additional info in output filename to describe experiments'
)
parser.add_argument(
    '--random_inputs',
    default=False,
    help="Randomize the inputs so that directions need to be learned initially",
    action='store_true'
)
parser.add_argument(
    '--expert_view',
    default=False,
    help="Draw the overall map along with the agent view. for debugging",
    action='store_true'
)


args = parser.parse_args()
env = gym.make(args.env)

# a settings.json file should be included for logging to comet.
# ----------------------------------------------------------------
if args.debug:
    args.comet = False # no logging when debugging.
    import ipdb
    ipdb.set_trace()


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


# Globals
episode_num = 0
last_time = time.time()
order_choices = np.arange(3)
save_name = args.namestr+"_run" # think about what name I want to save.
action_sequence = []
state_draw_sequence = []
state_position_sequence = []
decision_time_logs = []
reward_logs = []
key_press_sequence = []
# goal_position = env. hmm, it's set up to be in bottom right corner now.
# goal_position = []
state_action_table = {}


#state_draw_sequence.append(env.__str__)

if args.random_inputs:
    order_choices = np.random.permutation(np.arange(3))

if args.agent_view:
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)

window = Window('gym_minigrid - ' + args.env)
window.reg_key_handler(key_handler)

reset()

# Blocking event loop
window.show(block=True)
