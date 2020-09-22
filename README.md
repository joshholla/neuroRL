# Neuro RL correlates

Welcome!

This project aspires to uncover correlates between human brains and RL agents!  

We plan on doing this via having people play a gridworld game while connected to an fMRI machine.   
The gridworld is built upon the fine work at https://github.com/maximecb/gym-minigrid

## Getting started
First ensure that you have installed python3 on your system.   
Install the gym environment and other dependencies by navigating to this folder and running:
```
pip3 install -e .
```
(this should install the additional custom environments required for running our tests)   


If you would like to play the game, assuming you are in the root folder, run 

```
cd neuroRL
python enhanced_neuro_view.py
```
The following flags add functionality: 
```
--comet [Flag to toggle logging data to cometml]  
--namestr [Flag for the remote name of the experiment (in cometml)]  
--random_inputs [Including these inputs will randomize the inputs on the keyboard]  
--expert_view [This will let the user see the entire map including what the agent normally has vision of]  
```


For logging to comet.ml, please ensure that you have a settings.json file in the `/rlscripts` folder in the following format:
```
{
  "api_key": "<Insert Key from comet here>",
  "project_name": "<Where you'd like to log your results>",
  "workspace": "<usually your comet username>"
}
```

To run DQN on this environment, please run the following commands:
```
./runrl.sh
```

Thanks for visiting this project!


## TODO
> License?  
> good to go for BIC machines?  

