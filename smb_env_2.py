# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 19:48:02 2022

@author: db_wi
"""

import gym
from gym.spaces.tuple import Tuple
from gym.spaces.discrete import Discrete
from gym.spaces.box import Box
import gym_super_mario_bros as gsmb
import numpy as np

def clamp(n, smallest, largest): return max(smallest, min(n, largest))     

class SMBEnv2(gsmb.SuperMarioBrosEnv):
    HORIZONTAL_VELOCITY_ADDR = 0x0057
    VERTICAL_VELOCITY_ADDR = 0x009F
    reward_range = (-15.0, 15.0)
    # observation_space = Tuple((
    #     gsmb.SuperMarioBrosEnv.observation_space,
    #     Box(low=0,high=255,shape=(6,2), dtype=np.uint8),
    #     Box(low=np.array( ((0,0),) ),high=np.array( ((80,10),) ),shape=(1,2), dtype=np.uint8)
    #     ))
    observation_space = Box(low=0,high=255,shape=(2048,),dtype=np.uint8)
    
    # death should have a LOT more of a penalty than -15, but is this too much?
    # @property 
    # def _death_penalty(self):
    #     if self._is_dead or self._is_dying:
    #         return -150.0
    #     return 0.0


    # rewards should be a float, according to that irritating warning
    def _get_reward(self):
        return float(self._x_reward + self._time_penalty + self._death_penalty)
    @property
    def _player_velocity(self):
        ret = np.ndarray( (1,2), dtype='int8')
        ret[0] = (self.ram[self.HORIZONTAL_VELOCITY_ADDR]+40,
                    self.ram[self.VERTICAL_VELOCITY_ADDR]+5)
        return ret
    
    @property 
    def _player_screen_position(self):
        ret = np.ndarray( (1,2), dtype='uint8')
        ret[0] = (self.ram[0x3AD],self.ram[0x03B8])
        return ret
        
    @property
    def _enemy_screen_positions(self):
        ret = np.ndarray( (5,2), dtype='uint8')
        for i in range(5):
            if self.ram[0x000F + i] > 0:
                ret[i] = (self.ram[0x03AE + i],self.ram[0x03B9 + i])
            else:
                ret[i] = (0,0)
        return ret
                
    @property
    def _positions(self):
        return np.concatenate( (self._player_screen_position,self._enemy_screen_positions) )
        
    # add velocity information to step result
    # velocities are normalised relative to 128, since np.int8 causes
    # issues in GymWrapper and other parts of TensorFlow
    # def step(self,action):
    #     ret = super().step(action)
    #     ret =       ( 
    #                     (
    #                         ret[0],
    #                         self._positions,
    #                         self._player_velocity,
    #                     ),
    #                 ) + ret[1:]
    #     return ret
    def step(self,action):
        ret = super().step(action)
        return (self.ram,ret[1],bool(ret[2]),ret[3])
    
    def reset(self):
        ret = super().reset()
        return self.ram
        
    # add hitbox overlay
    def render(self,mode='human',**kwargs):
        i = 0
        for pos in self._enemy_screen_positions:
            if not (pos == 0).all():
                x1 = clamp(self.ram[0x04B0+4*i],0,255)
                y1 = clamp(self.ram[0x04B0+4*i+1],0,239)
                x2 = clamp(self.ram[0x04B0+4*i+2],0,255)
                y2 = clamp(self.ram[0x04B0+4*i+3],0,239)

                self.screen[clamp(pos[1],0,239)][clamp(pos[0],0,255)] = (255,255,0)
                for x in range(x1,x2+1):
                    c = 255 if x%4 > 1 else 0
                    self.screen[y1][x]= (c,c,c)
                    self.screen[y2][x]= (c,c,c)
                for y in range(y1,y2+1):
                    c = 255 if y%4 > 1 else 0
                    self.screen[y][x1]= (c,c,c)
                    self.screen[y][x2]= (c,c,c)
            i += 1
        pos = self._player_screen_position[0]
        self.screen[clamp(pos[1],0,239)][clamp(pos[0],0,255)] = (255,255,0)
        x1 = clamp(self.ram[0x04AC],0,255)
        y1 = clamp(self.ram[0x04AD],0,239)
        x2 = clamp(self.ram[0x04AE],0,255)
        y2 = clamp(self.ram[0x04AF],0,239)
        
        for x in range(x1,x2+1):
            c = 255 if x%4 > 1 else 0
            self.screen[y1][x]= (c,c,c)
            self.screen[y2][x]= (c,c,c)
        for y in range(y1,y2+1):
            c = 255 if y%4 > 1 else 0
            self.screen[y][x1]= (c,c,c)
            self.screen[y][x2]= (c,c,c)
        return super().render(mode)

_ID_TEMPLATE = 'smb-better{}-{}-{}-v{}'
_ROM_MODES = [
    'vanilla',
    'downsample',
    'pixel',
    'rectangle'
]
def _register_mario_stage_env(id, **kwargs):
    """
    Register a Super Mario Bros. (1/2) stage environment with OpenAI Gym.
    Args:
        id (str): id for the env to register
        kwargs (dict): keyword arguments for the SuperMarioBrosEnv initializer
    Returns:
        None
    """
    # register the environment
    gym.envs.registration.register(
        id=id,
        entry_point='smb_env_2:SMBEnv2',
        max_episode_steps=9999999,
        reward_threshold=9999999,
        kwargs=kwargs,
        nondeterministic=True,
    )
for version, rom_mode in enumerate(_ROM_MODES):
    for world in range(1, 9):
        for stage in range(1, 5):
            # create the target
            target = (world, stage)
            # setup the frame-skipping environment
            env_id = _ID_TEMPLATE.format('', world, stage, version)
            _register_mario_stage_env(env_id, rom_mode=rom_mode, target=target)
            
# register the new entry points 
gym.envs.registration.register(
    id='smb-better-v3',
    entry_point='smb_env_2:SMBEnv2',
    max_episode_steps=9999999,
    reward_threshold=9999999,
    kwargs = {'rom_mode':'rectangle'},
    nondeterministic=True
    )
gym.envs.registration.register(
    id='smb-better-vanilla-v3',
    entry_point='smb_env_2:SMBEnv2',
    max_episode_steps=9999999,
    reward_threshold=9999999,
    kwargs = {'rom_mode':'vanilla'},
    nondeterministic=True
    )
gym.envs.registration.register(
    id='smb-better-pixel-v3',
    entry_point='smb_env_2:SMBEnv2',
    max_episode_steps=9999999,
    reward_threshold=9999999,
    kwargs = {'rom_mode':'pixel'},
    nondeterministic=True
    )
gym.envs.registration.register(
    id='smb-better-downscaled-v3',
    entry_point='smb_env_2:SMBEnv2',
    max_episode_steps=9999999,
    reward_threshold=9999999,
    kwargs = {'rom_mode':'downscaled'},
    nondeterministic=True
    )
make = gym.make
