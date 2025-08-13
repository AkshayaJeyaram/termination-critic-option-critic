import logging
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

logger = logging.getLogger(__name__)


class GridFourRooms(gym.Env):
    """
    Custom GridWorld-style Four Rooms environment.
    The agent must navigate to a goal cell while avoiding walls.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        layout = """\
wwwwwwwwwwwww
w     w     w
w     w     w
w           w
w     w     w
w     w     w
ww wwww     w
w     www www
w     w     w
w     w     w
w           w
w     w     w
wwwwwwwwwwwww
"""
        self._grid = np.array(
            [list(map(lambda ch: 1 if ch == 'w' else 0, row))
             for row in layout.splitlines()]
        )

        self.action_space = spaces.Discrete(4)  # up, down, left, right
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(np.sum(self._grid == 0),)
        )

        self._directions = [np.array([-1, 0]), np.array([1, 0]),
                            np.array([0, -1]), np.array([0, 1])]

        self._rng = np.random.RandomState(1234)

        self._cell_to_index = {}
        self._index_to_cell = {}
        idx = 0
        for i in range(13):
            for j in range(13):
                if self._grid[i, j] == 0:
                    self._cell_to_index[(i, j)] = idx
                    self._index_to_cell[idx] = (i, j)
                    idx += 1

        self.goal_index = 62
        self.initial_state_space = list(range(self.observation_space.shape[0]))
        self.initial_state_space.remove(self.goal_index)

        self.current_position = None
        self._episode_steps = 0

    def seed(self, seed=None):
        return self._seed(seed)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _valid_moves(self, cell):
        moves = []
        for action in range(self.action_space.n):
            next_cell = tuple(cell + self._directions[action])
            if self._grid[next_cell] == 0:
                moves.append(next_cell)
        return moves

    def reset(self):
        state_idx = self._rng.choice(self.initial_state_space)
        self.current_position = self._index_to_cell[state_idx]
        self._episode_steps = 0
        return self._encode_state(state_idx)

    def switch_goal(self):
        previous_goal = self.goal_index
        self.goal_index = self._rng.choice(self.initial_state_space)
        self.initial_state_space.append(previous_goal)
        self.initial_state_space.remove(self.goal_index)
        assert previous_goal in self.initial_state_space
        assert self.goal_index not in self.initial_state_space

    def _encode_state(self, idx):
        state_vector = np.zeros(self.observation_space.shape[0])
        state_vector[idx] = 1
        return state_vector

    def render(self, show_goal=True):
        grid_copy = np.array(self._grid)
        x, y = self.current_position
        grid_copy[x, y] = -1
        if show_goal:
            gx, gy = self._index_to_cell[self.goal_index]
            grid_copy[gx, gy] = -1
        return grid_copy

    def step(self, action):
        self._episode_steps += 1
        next_pos = tuple(self.current_position + self._directions[action])

        if self._grid[next_pos] == 0:
            if self._rng.rand() < 1 / 3.:
                next_moves = self._valid_moves(self.current_position)
                self.current_position = next_moves[self._rng.randint(len(next_moves))]
            else:
                self.current_position = next_pos

        state_idx = self._cell_to_index[self.current_position]
        done = (state_idx == self.goal_index)
        reward = float(done)

        if not done and self._episode_steps >= 1000:
            done = True
            reward = 0.0

        return self._encode_state(state_idx), reward, done, None


if __name__ == "__main__":
    env = GridFourRooms()
    env.seed(3)
