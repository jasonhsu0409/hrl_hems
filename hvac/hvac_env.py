import pandas as pd
import numpy  as np
import gymnasium as gym
from gymnasium import spaces

class HVAC_env(gym.Env):
    def __init__(self, mode ='train'):
        super(HVAC_env, self).__init__()
        self.mode  = mode # train or test
        self.time_step  = None
        self.load_power = 4
        self.execute_price  = None
        self.load_use_power = None
        
        # Read data from csv
        self.T_set_data = None
        self.T_out_data = pd.read_csv("csv_data/TemperatureF.csv")
        self.price_data = pd.read_csv("csv_data/grid_price.csv") # ['summer_price'] ['not_summer_price']

        # execute data
        self.execute_remain_power = None
        self.execute_indoorT = None
        self.execute_outT = None

        self.month_column = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "July", "Aug", "Sep", "Oct", "Nov", "Dcb"]
       
        # Base Parameter
        self.PgridMax = 15
        self.epsilon  = 0.7
        self.eta      = 2.5
        self.A        = 0.14  #A(KW/F)
        self.initIndoorTemperature = 80

        # Observation space --------------------------------------------------------------------------------
        #[ time_step, remain_power, pgrid_price, indoorT, outdoorT, userSetT]
        self.low  = np.array([  0,    0,  0,  20,  35,  35], dtype=np.float32)
        self.high = np.array([ 95,  4.8,  7, 130, 130, 130], dtype=np.float32)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        # Ation space --------------------------------------------------------------------------------------
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

    def generate_remain_power(self):
        return np.random.rand(96)*self.load_power + np.random.randint(0,3,96)
    
    def generate_T_set_data(self):
        return np.ones(96)*77 + np.random.rand(96)*10 - 5
    
    ###########################################################################
    def step(self, action):
        # Get state -----------------------------------------------------------
        self.time_step, remain_power, price, T_in, T_out, T_set = self.state
        self.time_step = int(self.time_step)

        hvac_power = ( action[0] + 1 )*0.5 * remain_power

        self.load_use_power = hvac_power

        T_in_next = self.epsilon * T_in + ( 1 - self.epsilon )*( T_out - ( self.eta / self.A )* hvac_power/4*1.2 )
        self.T_in = T_in_next
        
        reward_cost = -1*hvac_power * price / ( 7 * self.load_power )

        if T_out < T_set :
            reward_preference = 0
            self.load_preference = 0 
        else:
            reward_preference = -1 * ( ( abs(T_in_next - T_set)+1 )**2 )/10
            self.load_preference = abs(T_in_next - T_set)

        self.reward = ( ( 3*reward_cost ) + ( 7*reward_preference ) ) /10 # pgridmax =2

        # 2 5 3

        self.time_step += 1

        if ( self.time_step >= 96 ) :
            done = True
            price_next = price
            T_out_next = T_out
            T_set_next = T_set
            remain_power_next = remain_power
        else:
            done = False
            price_next = self.execute_price_data[self.time_step]
            T_out_next = self.execute_T_out_data[self.time_step]
            T_set_next = self.execute_T_set_data[self.time_step]
            remain_power_next = min( self.load_power, self.execute_remain_power[self.time_step])

        self.state = np.array([ self.time_step, remain_power_next, price_next, T_in_next, T_out_next, T_set_next], dtype=np.float32)
        # truncated = False
        return  ( self.state, self.reward, done, False, {} )
    
    def reset(self, seed=None, options=None, input_month = 1): #id for choose HVAC, #month = 1-12
        super().reset(seed=seed)
        self.time_step  = 0 
        self.execute_remain_power = self.generate_remain_power()

        if self.mode == 'train':
            # self.month  = np.random.randint( 1, 13)
            self.month  = np.random.randint( 6, 10)
            price_noise = np.random.normal( 0, 0.1,96)
            T_out_noise = np.random.normal( 0,   1,96)
        else:
            self.month  = input_month 
            price_noise = np.zeros(96)
            T_out_noise = np.zeros(96)

        self.month_name = self.month_column[int(self.month - 1)]

        if ( 6 <= self.month <= 9 ): 
            self.execute_price_data = np.array(self.price_data['summer_price']) + price_noise
        else: 
            self.execute_price_data = np.array(self.price_data['not_summer_price']) + price_noise
        
        self.execute_T_out_data = self.T_out_data[self.month_name] + T_out_noise

        self.execute_T_set_data = pd.read_csv("csv_data/userSetTemperatureF.csv")[self.month_name] 

        if self.mode == 'train':
            self.execute_T_set_data = pd.read_csv("csv_data/userSetTemperatureF.csv")[self.month_column[int(np.random.randint(0,12))]] + np.random.normal( 0, 0.5, 96)
        else:
            self.execute_T_set_data = pd.read_csv("csv_data/userSetTemperatureF.csv")[self.month_name] 
        #------------------------------------------------------------------------------
        price_next = self.execute_price_data[self.time_step] 
        T_out_next = self.execute_T_out_data[self.time_step]
        T_set_next = self.execute_T_set_data[self.time_step]
        T_in_next  = np.random.randint(70,81)
        self.T_in = T_in_next
        
        remain_power_next = min( self.load_power, self.execute_remain_power[self.time_step] )

        self.state = np.array([ self.time_step, remain_power_next, price_next, T_in_next, T_out_next, T_set_next], dtype=np.float32)

        return ( self.state, {} )

    def render(self):
        pass