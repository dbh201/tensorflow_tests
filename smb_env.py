# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 19:00:52 2022

@author: db_wi
"""
import gym
from gym.spaces.tuple import Tuple
from gym.spaces.discrete import Discrete
from gym.spaces.box import Box
import gym_super_mario_bros as gsmb
import numpy as np
        
class SMBEnv2(gsmb.SuperMarioBrosEnv):
    HORIZONTAL_VELOCITY_ADDR = 0x0057
    VERTICAL_VELOCITY_ADDR = 0x009F
    reward_range = (-15.0, 15.0)
    observation_space = Tuple((
        gsmb.SuperMarioBrosEnv.observation_space,
        Box(low=np.array([[0,0]]),high=np.array([[80,10]]),shape=(1,2), dtype=np.uint8)
        ))
    # rewards should be a float, according to that irritating warning
    def _get_reward(self):
        return float(super()._get_reward())
    
    # add velocity information to step result
    # velocities are normalised relative to 128, since np.int8 causes
    # issues in GymWrapper and other parts of TensorFlow
    def step(self,action):
        ret = super().step(action)
        ret =       ( 
                        (
                            ret[0],
                            ((
                            (self.ram[self.HORIZONTAL_VELOCITY_ADDR] + 40) % 256,
                            (self.ram[self.VERTICAL_VELOCITY_ADDR] + 5) % 256
                            ),),
                        ),
                    ) + ret[1:]
        return ret
    def reset(self):
        ret = super().reset()
        return      ret, (((self.ram[self.HORIZONTAL_VELOCITY_ADDR] + 40) % 256,\
                    (self.ram[self.VERTICAL_VELOCITY_ADDR] + 5) % 256),)
        
# register the new entry points 
gym.envs.registration.register(
    id='smb-better-v3',
    entry_point='smb_env:SMBEnv2',
    max_episode_steps=9999999,
    reward_threshold=9999999,
    kwargs = {'rom_mode':'rectangle'},
    nondeterministic=True
    )
gym.envs.registration.register(
    id='smb-better-pixel-v3',
    entry_point='smb_env:SMBEnv2',
    max_episode_steps=9999999,
    reward_threshold=9999999,
    kwargs = {'rom_mode':'pixel'},
    nondeterministic=True
    )
gym.envs.registration.register(
    id='smb-better-downscaled-v3',
    entry_point='smb_env:SMBEnv2',
    max_episode_steps=9999999,
    reward_threshold=9999999,
    kwargs = {'rom_mode':'downscaled'},
    nondeterministic=True
    )
make = gym.make
