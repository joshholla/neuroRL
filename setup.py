from setuptools import setup

setup(
    name='neuroRL',
    version='0.1.0',
    keywords='game, environment, agent, rl, gym, gridworld, minigrid',
    url='https://github.com/joshholla/neuroRL',
    description='Minimalistic gridworld game. Built for testing RL agents and human beings',
    packages=['neuroRL', 'neuroRL.envs'],
    install_requires=[
        'gym>=0.9.6',
        'numpy>=1.15.0',
        'gym_minigrid==1.0.1',
        'comet-ml==3.2.1',
        'matplotlib>=3.3.2'
    ]
)
