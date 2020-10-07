# Neuro RL correlates

Welcome!

This project aspires to uncover correlates between human brains and RL agents!  
We invite people play our gridworld game while an fMRI machine measures brain activity.  

The gridworld environment is built upon the fine work at https://github.com/maximecb/gym-minigrid

## Getting started
First ensure that you have installed python3 on your system.   

Begin by cloning this project. 
```
git clone https://github.com/joshholla/neuroRL.git 
```

The folder locally structure should look like this

```
.
`-- neuroRL
    |-- README.md
    |-- neuroRL
    |   |-- __init__.py
    |   |-- enhanced_neuro_view.py
    |   |-- envs
    |   |   |-- __init__.py
    |   |   `-- empty.py
    |   |-- rlscripts
    |   |   |-- DQN.py
    |   |   |-- main.py
    |   |   |-- utils.py
    |   |   `-- visualize_Pending.py
    |   |-- runrl.sh
    |   `-- utils.py
    `-- setup.py
```
To set up a virtual environment, run the following commands
```
cd neuroRL
virtualenv --system-site-packages -p python3.6 neuroRLenv
source neuroRLenv/bin/activate
```


Conda users can alternatively use these commands:
```
cd neuroRL
conda create --name neuroRLenv python=3.6
conda activate neuroRLenv

```

Install the gym environment and other dependencies by navigating to this folder and running:
```
pip3 install -e .
```
(this will install the additional custom environments required for running our tests)  

If you would like to play the game, first ensure that you activate the virtual environment we created using either of the two commands below, depending on which tool you used:
```
source neuroRLenv/bin/activate
conda activate neuroRLenv
```
Assuming you are in the folder where this README resides, run 

```
cd neuroRL
python enhanced_neuro_view.py
```

To control the agent in the gridworld, use the following buttons on your keyboard:

- "1" to pivot left.
- "2" to move the agent forward.
- "3" to pivot right.
We wanted to account for bias in directional arrow keys and so we've chosen these buttons for this version.  
Future versions plan on connecting to a game controller for movement and interaction.  

The following flags add functionality to your experiments: 
```
--comet [Flag to toggle logging data to cometml]  
--namestr [Flag for the remote name of the experiment (in cometml)]  
--random_inputs [Including these inputs will randomize the inputs on the keyboard]  
--expert_view [Enables access to higher debugging and an overview of agent activity]  
```

For logging to comet.ml, please ensure that you have a `settings.json` file in the `/rlscripts` folder in the following format:
```
{
  "api_key": "<Insert Key from comet here>",
  "project_name": "<Where you'd like to log your results>",
  "workspace": "<usually your comet username>"
}
```
If we've sent you an email asking for help, a `settings.json` file will be attached along with that email.


To run an RL agent on this environment, you can run the following commands:
```
./runrl.sh
```

If you're able to run this project locally, and have played around with the game, we have a few questions for you!
```
+ Does the game work for you? did you run into any issues with getting it to run?
+ Do you think it is suitable for an fMRI environment?
+ What could improve the experience?
+ Is it engaging? How long did you play for?
+ Would you change the controls to anything else?
```
Do send your feedback to neurorlfeedback[at]gmail.com



Thanks for visiting this project!

