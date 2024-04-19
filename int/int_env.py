import pandas as pd
import numpy  as np
import gymnasium as gym
from gymnasium import spaces
from sb3_contrib.common.maskable.utils import get_action_masks

class IntEnv(gym.Env):
    def __init__(self, mode='train'):
        super(IntEnv, self).__init__()
        self.mode = mode # train or test
        self.load_power  = None
        self.time_step   = None
        self.load_demand = None
        self.load_remain_demand = None
        self.execute_price      = None
        self.execute_preference = None
        self.load_use_power = None
        self.load_preference = None
        self.reward = None
        
        # read price data
        self.price_data = pd.read_csv("csv_data/grid_price.csv")#['summer_price'] ['not_summer_price']

        # read preference data
        self.preference_data = {
            1:pd.read_csv("csv_data/intPreference1.csv"),
            2:pd.read_csv("csv_data/intPreference2.csv"),
            3:pd.read_csv("csv_data/intPreference3.csv")
        }
        
        # Observation space --------------------------------------------------------------------------------
        #[ time_step, pgrid_price, remain_time, preference]
        lowerLimit = np.array([  0,  0,  0, -4 ], dtype=np.float32)
        upperLimit = np.array([ 95,  7, 40,  4 ], dtype=np.float32)
        self.observation_space = spaces.Box(lowerLimit, upperLimit, dtype=np.float32)

        # Ation space --------------------------------------------------------------------------------------
        self.action_space = spaces.Discrete(2)

    ###########################################################################
    def step(self, action):
        # Get state -----------------------------------------------------------
        #[ time_step, remain_power, pgrid_price,  remain_time, preference]
        self.time_step, Pgrid_price, state_remain, preference = self.state
        self.time_step = int(self.time_step)
    
        reward_cost = 0
        reward_preference  = 0
        not_enough_penalty = 0

        if (action == 1)and(self.load_remain_demand>0):
            self.load_remain_demand -= 1
            reward_cost       = ( Pgrid_price / 7 ) / self.load_demand
            reward_preference = ( preference  / 4 ) / self.load_demand
            self.load_is_on   = 1
        else:
            self.load_is_on   = 0

        self.load_use_power = self.load_is_on*self.load_power

        self.load_preference = preference

        self.time_step += 1 

        if ( self.time_step >= 96 ) :
            done = True
            if  self.load_remain_demand > 0:
                not_enough_penalty = -1
            else:
                not_enough_penalty =  1
        else:
            done = False

        if ( self.time_step >= 96 ):
            price_next      = Pgrid_price
            preference_next = preference
        else:
            price_next      = self.execute_price[self.time_step]
            preference_next = self.execute_preference[self.time_step]

        # reward = ( -0.7*reward_cost + 0.3*reward_preference ) + 20*not_enough_penalty
        self.reward = (  -3*reward_cost + 7*reward_preference  + 20*not_enough_penalty ) /30
        
        # 1 cost , 5 target, 2 preference, 2 pgridmax

        # Next state ----------------------------------------------------------
        self.state = np.array([ self.time_step, price_next, self.load_remain_demand, preference_next ], dtype=np.float32)

        # truncated = False
        return  ( self.state, self.reward, done, False, {} )
    
    ###########################################################################
    def reset(self, seed=None, options=None, id=1, month=1, demand = 30, load_power=1.5 ):# id = 1,2,3 / month = 1 ~ 12
        super().reset(seed=seed)
        self.time_step = 0
        if self.mode == 'train':
            self.id = np.random.randint(1,4)
            self.month = np.random.randint()
            self.load_power  = load_power
            self.load_demand = np.random.randint(20,26)
            price_noise      = np.random.normal(0,0.1,96)
            preference_noise = np.random.randint(-1,2,96)
        else:
            self.id = id
            self.month = month
            self.load_power  = load_power
            self.load_demand = demand
            price_noise      = np.zeros(96)
            preference_noise = np.zeros(96)

        self.load_remain_demand = self.load_demand

        # get preference_data
        self.execute_preference = self.preference_data[self.id][str(self.month)] + preference_noise
        for i in range(len(self.execute_preference)):
            if self.execute_preference[i] > 4:
                self.execute_preference[i] = 4
            elif  self.execute_preference[i] < -1:
                self.execute_preference[i] = -1


        self.execute_price = self.price_data['summer_price'] + price_noise

        #------------------------------------------------------------------------------
        #[ time_step, remain_power, pgrid_price,  Remain_time, preference]
        price_next      = self.execute_price[self.time_step]
        preference_next = self.execute_preference[self.time_step]

        self.state = np.array([ self.time_step,  price_next, self.load_remain_demand, preference_next ], dtype=np.float32)

        return ( self.state, {} )

    def render(self):
        pass