import pandas as pd
import numpy  as np
import gymnasium as gym
from gymnasium import spaces
from sb3_contrib.common.maskable.utils import get_action_masks

class UnintEnv(gym.Env):
    def __init__(self, mode='train'):
        super(UnintEnv, self).__init__()
        self.mode = mode # train or test
        self.time_step = None
        self.execute_price = None
        self.execute_preference = None
        self.execute_remain_power = None
        self.preference_data_index = 0

        # load fix variable
        self.plt_load   = None
        self.load_power = None
        self.load_demand = None
        self.load_period = None

        # load variable
        self.load_switch = None
        self.load_remain_demand = None
        self.load_remain_period = None
        self.load_use_power = None
        
        # read price data
        self.price_data = pd.read_csv("csv_data/grid_price.csv")

        # read preference data

        self.preference_data = {
            1 : pd.read_csv("csv_data/unIntPreference1.csv"),
            2 : pd.read_csv("csv_data/unIntPreference2.csv")
        }

        # Observation space --------------------------------------------------------------------------------
        # [ time_step,, pgrid_price, remain_time, remain_period, period_len, preference ]
        lowerLimit = np.array([  0,  0,  0,  0, 0, -4 ], dtype=np.float32)
        upperLimit = np.array([ 95,  7, 40,  7, 7,  4 ], dtype=np.float32)
        self.observation_space = spaces.Box(lowerLimit, upperLimit, dtype=np.float32)

        # Ation space --------------------------------------------------------------------------------------
        self.action_space =  spaces.Discrete(2)
 
    ###########################################################################
    def step(self, action):
        # Get state -----------------------------------------------------------
        # [ time_step,, pgrid_price, remain_time, remain_period, period_len, preference ]
        self.time_step,  Pgrid_price, state_remain, _, _, preference = self.state
        
        not_enough_penalty = 0
        cost_reward        = 0
        preference_reward  = 0

        if ( ( action == 1 ) & ( self.load_remain_period == 0 ) & ( self.load_remain_demand > 0 ) ):
            self.load_remain_period = self.load_period
        
        if self.load_remain_period > 0:
            self.load_is_on = 1
            self.load_remain_demand -= 1
            self.load_remain_period -= 1
            cost_reward       = ( Pgrid_price/7 ) / ( self.load_demand )
            preference_reward = ( preference /4 ) / ( self.load_demand )
        else:
            self.load_is_on   = 0
            cost_reward       = 0
            preference_reward = 0

        self.load_use_power  = self.load_is_on*self.load_power

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
            price_next        = Pgrid_price
            preference_next   = preference
        else:
            price_next        = self.execute_price[self.time_step]
            preference_next   = self.execute_preference[self.time_step]

        self.reward = ( ( -5*cost_reward + 5*preference_reward ) + 20*not_enough_penalty)/30

        # 2 cost , 5 target, 2 preference, 1 pgridmax

        if  ( state_remain == 0 ) & (self.mode == 'train' ) & ( self.time_step < 96 ) :
            if np.random.randint(10) > 5:
                max_num_of_period = max( 1, ( ( 96 - self.time_step)//self.load_period ) - 1) 
                self.load_remain_demand = self.load_period*( np.random.randint( 0, max( 1, max_num_of_period )) )
    
        # Next state ----------------------------------------------------------
        # [ time_step,, pgrid_price, remain_time, remain_period, period_len, preference ]
        self.state = np.array([ self.time_step, price_next, self.load_remain_demand,  self.load_remain_period, self.load_period, preference_next ], dtype=np.float32)

        # truncated = False
        return  ( self.state, self.reward, done, False, {} )
    
    ###########################################################################
    def reset(self, seed=None, options=None, id=1, month=6, demand = 30, period = 5, load_power = 1.5):# id = 1,2 / month = 1 ~ 12
        super().reset(seed=seed)
        self.time_step  = 0
        self.load_power = load_power

        # get preference data
        if self.mode == 'train': # testing mode read the id and month
            self.id    = np.random.randint(1, 3)
            # self.month = np.random.randint(1,13)
            self.month = np.random.randint(6,10)
            self.load_period = np.random.randint(5,8)
            self.load_demand = (np.random.randint(30,36)//self.load_period)*self.load_period
            price_noise      = np.random.normal(0,0.1,96)
            preference_noise = np.random.randint(-1,2,96)
            self.execute_preference = self.preference_data[self.id][str(self.month)] + preference_noise
        else:
            self.id    = id
            self.month = month
            self.load_period = period
            self.load_demand = demand
            price_noise      = np.zeros(96)
            preference_noise = np.zeros(96)
            self.execute_preference = self.preference_data[self.id][str(self.month)]

        self.load_remain_demand = self.load_demand
        self.load_remain_period = 0
        
        
        # whether is "summer" | 6-9
        if ( 6 <= self.month <= 9 ): # is summer
            self.execute_price = self.price_data['summer_price'] + price_noise
        else: # not summer
            self.execute_price = self.price_data['not_summer_price'] + price_noise

        #------------------------------------------------------------------------------
        price_next        = self.execute_price[self.time_step]
        preference_next   = self.execute_preference[self.time_step]

        # if preference_next < 0:
        #     preference_next = -4

        # [ time_step,, pgrid_price, remain_time, remain_period, period_len, preference ]
        self.state = np.array([ self.time_step, price_next, self.load_remain_demand, self.load_remain_period, self.load_period, preference_next ], dtype=np.float32)

        return ( self.state, {} )

    def render(self):
        pass