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

The folder structure should look like this:
```
.  <--- (YOU ARE HERE! :)
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

Now its time to set up a virtual evironment. I prefer using conda.  

To set up a virtual environment using virtualenv, run the following commands
```
cd neuroRL
virtualenv --system-site-packages -p python3.6 neuroRLenv
source neuroRLenv/bin/activate
```

Miniconda or conda users can alternatively use these commands:
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

## Controls:
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


## To run experiments:

If you would like to play the game, first ensure that you activate the virtual environment we created using either of the two commands below, depending on which tool you used:
```
source neuroRLenv/bin/activate
conda activate neuroRLenv
```

Assuming you are in the folder where this README resides, run 

```
cd neuroRL
```
Alright, let's set up some variables that will help with logging:  
Ensure you've copied over the `settings.json` file over to the `\rlscripts` folder (the `settings.json` file should be attached with the email that was sent to you OR if you've stumbled on one, feel free to make your own settings.json file with the format listed above - populating it with your relevant comet credentials)  
This bit is important and will need your input to update each time you run an experiment.
```
local mynamestr="<unique name here. I suggest writing your name with today's date and a number eg - josh11Jan2021_1>"
```

Now you can launch the project using any the following commands (Try them out a few times, hopefully you'll have fun!)  
Note that you can end a session by pressing `q`  
Running this will laungh the game with the default controls as listed above:
```
python enhanced_neuro_view.py --comet --namestr=$mynamestr
```

Running this will randomize the inputs (the keys are still the same but are mapped to different actions)  
(don't forget to change `$mynamestr` if you've already run an experiment) 
```
python enhanced_neuro_view.py --comet --namestr=$mynamestr --random_inputs
```

If you'd like to launch the game without any logging, you can use this:
```
python enhanced_neuro_view.py --comet --namestr=$mynamestr
```

## Feedback:  
If you're able to run this project locally, and have played around with the game, we have a few questions for you!  
(This can be entered in the google form that was mailed to you.)  
Read on if you stumbled upon this project:  
```
+ Does the game work for you? did you run into any issues with getting it to run?
+ Do you think it is suitable for an fMRI environment?
+ What could improve the experience?
+ Is it engaging? How long did you play for?
+ Would you change the controls to anything else?
```

If running this project resulted in errors on the console, I'd love to hear about them (so that I can fix em!)  

Do send your feedback or error messages to neurorlfeedback[at]gmail.com  



Thanks for visiting this project!

