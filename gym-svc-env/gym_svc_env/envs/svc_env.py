import gym
from gym import error, spaces, logger, utils
from gym.utils import seeding
import numpy as np
import math

class SvcEnv(gym.Env):
  """
  Description:
      The Svc environment is a room represented by a matrix(100x100). Each cell represent an area of 400cm^2 (20x20cm). A total of 400m^2(20x20m).
      The cell can have th following values:
      0 --> free and clean space;
      1 --> free and low dirty
      2 --> free and high dirty
      3 --> Blocked space

      The robot can start in any cell. And the objective is clean the maximun area using less movements.

  Source:
      This environment corresponds to the version of room by Everton Lima Aleixo

  Observation: 
      Type: Box(4)
      Num	Observation                 
      0	Front vision                  a sub space of universe (5x5)
      1	Left vision                   a sub space of universe (5x5)
      2	Right vision                  a sub space of universe (5x5)
      3	Back vision                   a sub space of universe (5x5)
      
  Actions:
      Type: Discrete(5)
      Num	Action
      0	Move to Front
      1	Move to Left
      2	Move to Right
      3	Move to Back
      4	Stop
      
  Reward:
      Reward is 1 when the action taken clean a square and -0.2 when not. If the vacuum cleaner choose stop action the episode ends 
      and the last reward will be (-1*'free and low dirty cells).

  Starting State:
      The robot can start in any cell.

  Episode Termination:
      Take an action that go to a blocked space
      Episode length is greater than 20000
      Solved Requirements
      travelled by all areas
  """

  metadata = {'render.modes': ['human']}

  def __init__(self):
    self.seed()
    self.MAX_SIZE_ROOM = 99
    self.done = False

    self.state = np.zeros((5,5,4), dtype=int)    
    self.reset()
    self.action_space = spaces.Discrete(5)
    self.observation_space = np.array((5,5,4), dtype=int)



  def step(self, action):
    assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
    if self.done:
      logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
      return self.state, 0, True, {}

    if action == 4: # Decide to stop
      self.done = True
      return self.state, -np.sum(self.universe==1), True, {}
    
    next_position = self.position.copy()
    if action == 0: # Move to front
      next_position[0] =  self.position[0] - 1
      next_position[1] =  self.position[1]
    elif action == 1: # Move to left
      next_position[0] =  self.position[0]
      next_position[1] =  self.position[1] - 1
    elif action == 2: # Move to right
      next_position[0] =  self.position[0]
      next_position[1] =  self.position[1] + 1
    elif action == 3: # Move to back
      next_position[0] =  self.position[0] + 1
      next_position[1] =  self.position[1]

    self.position = next_position
    # Check the new cell
    # Collision?
    if self.universe[next_position[0], next_position[1]] == 3:
      self.done = True
      return self.state, -np.sum(self.universe==1), True, {}

    reward = 1.0 if self.universe[next_position[0], next_position[1]] == 1 else (-0.2+self.visited[next_position[0], next_position[1]])
    
    if self.universe[next_position[0], next_position[1]]==1:
      self.universe[next_position[0], next_position[1]] = 0

    # Mark as visited. It will be punished next time pass here 
    self.visited[next_position[0], next_position[1]] = self.visited[next_position[0], next_position[1]] + 0.1

    self.update_state()
    
    return self.state, reward, False, {}
  
  def reset(self):
    self.universe = np.zeros((self.MAX_SIZE_ROOM + 1,self.MAX_SIZE_ROOM + 1))
    self.visited = np.zeros((self.MAX_SIZE_ROOM + 1,self.MAX_SIZE_ROOM + 1))
    self.total_movements = 0
    self.done = False

    self.create_universe()
    self.position = np.array([np.random.randint(self.MAX_SIZE_ROOM + 1), np.random.randint(self.MAX_SIZE_ROOM + 1)])
    while self.universe[self.position[0]][self.position[1]] == 3:
      self.position = np.array([np.random.randint(self.MAX_SIZE_ROOM + 1), np.random.randint(self.MAX_SIZE_ROOM + 1)])
    
    self.visited[self.position[0]][self.position[1]] = 1
    self.update_state()

    return self.state

  def render(self, mode='human', close=False):
    x = self.universe.copy()
    x[self.position[0]][self.position[1]] = 9
    print(x)
    for i in range(4): 
      print(self.state[:,:,i]) 
      print()

  def update_state(self):
    x = self.position[0]
    y = self.position[1]
    self.state[:,:,:] = 3

    # Top View
    aux = self.universe[np.maximum(0, x-5):x, np.maximum(0, y-2):np.minimum(self.MAX_SIZE_ROOM+1, y+3)]
    left_col = np.maximum(0, -(y-2))
    rigth_col = (5 - np.maximum(0, -(self.MAX_SIZE_ROOM+1-(y+3)) ))
    self.state[5-aux.shape[0]: , left_col:rigth_col ,0] = aux

    # Left View
    aux = self.universe[np.maximum(0, x-2):np.minimum(self.MAX_SIZE_ROOM+1, x+3), np.maximum(0, y-5):y ]
    top_lin = np.maximum(0, -(x-2))
    botton_lin = (5 - np.maximum(0, -(self.MAX_SIZE_ROOM+1-(x+3)) ))
    self.state[top_lin:botton_lin , 5-aux.shape[1]: , 1] = aux

    # Right View
    top_lin = np.maximum(0, -(x-2))
    botton_lin = (5 - np.maximum(0, -(self.MAX_SIZE_ROOM+1-(x+3)) ))
    aux = self.universe[np.maximum(0, x-2):np.minimum(self.MAX_SIZE_ROOM+1, x+3), y+1:np.minimum(self.MAX_SIZE_ROOM+1, y+5)]
    self.state[top_lin:botton_lin , :aux.shape[1] , 2] = aux

    # Botton View
    aux = self.universe[np.maximum(x+1, self.MAX_SIZE_ROOM):np.minimum(self.MAX_SIZE_ROOM+1, x+5):, np.maximum(0, y-2):np.minimum(self.MAX_SIZE_ROOM+1, y+3)]
    left_col = np.maximum(0, -(y-2))
    rigth_col = (5 - np.maximum(0, -(self.MAX_SIZE_ROOM+1-(y+3)) ))
    self.state[:aux.shape[0], left_col:rigth_col ,3] = aux

  def create_universe(self):
    for i in range(0,self.MAX_SIZE_ROOM + 1, 1):
      self.universe[i][0] = 3
      self.universe[i][self.MAX_SIZE_ROOM] = 3
      self.universe[0][i] = 3
      self.universe[self.MAX_SIZE_ROOM][i] = 3

    for i in range(1,self.MAX_SIZE_ROOM, 1):
      for j in range(1,self.MAX_SIZE_ROOM, 1):
        random_val = np.random.normal()
        cell_val = 1 #free and low dirty

        if random_val < -1:
          cell_val = 3 # Blocked
        if random_val > 1:
          cell_val = 0 # Free but already clean
        if random_val > 2:
          cell_val = 2 # Free and high dirty
        
        self.universe[i][j] = cell_val


  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]
