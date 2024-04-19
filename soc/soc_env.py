import pandas as pd
import numpy  as np
import gymnasium as gym
from gymnasium import spaces

class SocEnv(gym.Env):
    def __init__(self, mode ='train'):
        super(SocEnv, self).__init__()
        # env
        self.mode = mode
        self.Pgrid_max = 15
        self.month_name_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dcb']

        # soc
        self.soc = None
        self.batteryCapacity = 6
        self.soc_inital = 0.6
        self.soc_target = 0.6
        self.time_step  = None
        self.load_power = None
        self.execute_price = None

        # data
        self.price_data  = pd.read_csv("csv_data/grid_price.csv")   # ['summer_price'] ['not_summer_price']
        self.pv_data     = pd.read_csv("csv_data/PhotoVoltaic.csv") # ['month']

        if self.mode =='train':
            self.fixload = pd.read_csv("csv_data/TrainingData.csv") # ['day1_powerConsumption']
        else:
            self.fixload = pd.read_csv("csv_data/TestingData.csv") 

        # Observation space --------------------------------------------------------------------------------
        #[ time_step, remian_power, soc, pgrid_price]
        self.low  = np.array([  0, -10, 0.0, 0.0], dtype = np.float32)
        self.high = np.array([ 96,  20, 1.0, 7.0], dtype = np.float32)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        # Action space
        self.action_space = gym.spaces.Box( low=-1, high=1, shape=(1,))

    def generate_remain_power(self):
        # return np.random.rand(96)*15 + np.random.normal( 0, 0.2, 96)
        return ( np.random.randint(2,13,96) + np.random.normal(0,0.4,96) )
    
    ###########################################################################
    def step(self, action):
        # Get state -----------------------------------------------------------
        self.time_step, unc_load_power, soc, price = self.state

        # Reset reward --------------------------------------------------------
        reward_over_bound = 0
        reward_Pgrid_max = 0
        reward_cost = 0
        reward_target = 0
        
        # action > 0 charge 
        delta_soc = 0.3*action[0]
        soc_next  = soc + delta_soc

        # action > 0 charge | action < 0 discharge
        if delta_soc > 0:
            eta = 1/0.9
        else:
            eta = 0.9

        if soc_next > 1 :
            soc_next = 1
            reward_over_bound = -1

        elif soc_next < 0 :
            soc_next = 0
            reward_over_bound = -1

        self.soc = soc_next

        soc_use_power = (soc_next - soc)*self.batteryCapacity*eta

        self.load_use_power = soc_use_power

        if soc_use_power + unc_load_power > self.Pgrid_max :
            reward_Pgrid_max = -1

        # Pgrid = max(0 , (soc_use_power + unc_load_power ))

        reward_cost = -0.2*price * soc_use_power

        self.time_step = int(self.time_step + 1)

        if self.time_step < 96:
            next_remain_power = self.remain_power[self.time_step]
            next_price = self.execute_price[self.time_step]
            self.done = False

        else:
            next_remain_power = unc_load_power
            next_price = price
            self.done = True

            if( soc_next < self.soc_target ):
                reward_target = -1
            else:
                reward_target = 1


        self.reward = (2*reward_cost + reward_over_bound + 5*reward_Pgrid_max + 2*reward_target)  / 10

        self.state = np.array([ self.time_step, next_remain_power, soc_next, next_price ], dtype = np.float32)

        # truncated = False
        return  ( self.state, self.reward, self.done, False, {} )
    
    def reset(self, seed=None, day = 1):
        super().reset(seed=seed)
        self.time_step = 0
        self.remain_power = self.generate_remain_power()

        if  self.mode =='train':
            price_noise = np.random.normal( 0, 0.1, 96)
            self.day    = np.random.choice(np.arange(1,360))
            self.month  = self.day//30 + 1
        else:
            price_noise = np.zeros(96)
            self.day    = day
            self.month  = day

        self.month_name_list

        self.execute_price = self.price_data['summer_price'] + price_noise

        soc_next = self.soc_inital + np.random.normal(0,0.05)
        self.soc = self.soc_inital

        #------------------------------------------------------------------------
        self.state = np.array([ self.time_step, self.remain_power[self.time_step], soc_next, self.execute_price[self.time_step]], dtype = np.float32)

        return( self.state, {})

    def render(self):
        pass