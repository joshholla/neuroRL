from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class EmptyEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(
        self,
        size=8,
        agent_start_pos=(1,1),
        agent_start_dir=0,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        super().__init__(
            grid_size=size,
            max_steps=4*size*size,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()
            # tryingn to align the initial directions.
            self.agent_dir = self.agent_start_dir

        self.mission = "get to the green goal square"


class EmptyRandomEnv10x10(EmptyEnv):
    def __init__(self):
        super().__init__(size=10, agent_start_pos=None)

class EmptyEnv10x10(EmptyEnv):
    def __init__(self):
        super().__init__(size=10)

register(
    id='MiniGrid-Empty-Random-10x10-v0',
    entry_point='neuroRL.envs:EmptyRandomEnv10x10'
)

register(
    id='MiniGrid-Empty-10x10-v0',
    entry_point='neuroRL.envs:EmptyEnv10x10'
)

