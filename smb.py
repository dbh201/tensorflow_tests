# -*- coding: utf-8 -*-
"""
Created on Tue May 31 19:53:55 2022

@author: db_wi
"""

print("Importing gym...")
import gym_super_mario_bros as gsmb
print("    actions.COMPLEX_MOVEMENT")
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
print("Importing tensorflow...")
import tensorflow as tf
print("Importing numpy...")
import numpy as np
print("Importing keras-rl...")
import rl
print("    policy")
import rl.policy
print("    memory")
import rl.memory
print("    agents")
import rl.agents

from random import randint
print("Importing nes_py...\n    wrappers.JoypadSpace")
from nes_py.wrappers import JoypadSpace

from threading import Thread,Lock
from time import sleep
SAVE_LOCK = Lock()
# min and max number of frames to commit to an action
MAX_COMMITMENT = 10
MIN_COMMITMENT = 1

# WHAT DOTH A DEATH BEQUEATH THE ELECTRONIC FOOL???
DEATH_REWARD = -1000
DEATH_REWIND_FRAMES = 60

# Good things that can happen
COIN_REWARD = 10
SIZE_REWARD = 50


X_DATA_FILENAME = "X_DATA.npy"
NP_X_DATA = None
Y_DATA_FILENAME = "Y_DATA.npy"
NP_Y_DATA = None

X_FIELD_DATA_FILENAME = "X_FIELD.npy"
NP_X_FIELD_DATA = None
Y_FIELD_DATA_FILENAME = "Y_FIELD.npy"
NP_Y_FIELD_DATA = None


def status_to_int(x):
    return 0 if x == 'small' else 1 if x == 'tall' else 2
        
def run_until_done(tid):
    print("Running TID: %i" % tid)
    env = gsmb.make('SuperMarioBros-v3',disable_env_checker=True)
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    X_DATA = []
    Y_DATA = []
    done = False
    reward_sum = 0
    state = env.reset()
    action_plan = [ 0, 0 ]
    try:
        state, reward, done, info = env.step(action_plan[0])
        X_DATA.append([info,0,1])
        reward_sum = reward
        while not done:
            if action_plan[1] == 0:
                cr = 0
                cd = 0
                if X_DATA[-1][0]['life'] > info['life']:
                # The last few decisions led to a death.
                    Y_DATA.append(DEATH_REWARD)
                    dr = DEATH_REWIND_FRAMES
                    i = 1
                    while dr > 0:
                        try:
                            Y_DATA[-i] = DEATH_REWARD
                            dr -= X_DATA[-i][5]
                        except IndexError:
                            dr = 0
                else:
                    cr = status_to_int(info['status']) - status_to_int(X_DATA[-1][0]['status'])
                    cd = status_to_int(info['coins']) - status_to_int(X_DATA[-1][0]['coins'])
                  
                    Y_DATA.append(reward_sum + ( COIN_REWARD * cd) +
                                          (SIZE_REWARD * cr) )
                reward_sum = 0
                action_plan[0] = env.action_space.sample()
                action_plan[1] = randint(MIN_COMMITMENT,MAX_COMMITMENT)
                X_DATA.append([info,action_plan[0],action_plan[1]])
                
            state, reward, done, info = env.step(action_plan[0])
            reward_sum += reward
            action_plan[1] -= 1
    except KeyboardInterrupt:
        if len(X_DATA) > len(Y_DATA):
            X_DATA.pop()
    finally:
        if len(X_DATA) > len(Y_DATA):
            X_DATA.pop()
        print("Saving TID: %i which generated %i:%i records" % (tid,len(X_DATA),len(Y_DATA)))
        save(X_DATA,Y_DATA)
        env.close()

def convert(xdata,ydata):
    xdata2 = np.array([
            [ 
                x[0]['x_pos'],
                x[0]['y_pos'],
                x[0]['world'],
                x[0]['stage'],
                x[1],
                x[2] 
            ]
        for x in xdata ])
    ydata2 =  np.array(ydata, dtype=np.int16)
    return xdata2,ydata2
def save(xdata,ydata,xfilename=X_DATA_FILENAME,yfilename=Y_DATA_FILENAME):
    SAVE_LOCK.acquire()
    xdata2,ydata2 = convert(xdata,ydata)
    
    try:
        with open(xfilename,'rb') as xfile:    
            xdata3 = np.concatenate( (np.load(xfile),xdata2) )
    except FileNotFoundError:
        xdata3 = xdata2
    finally:
        np.save(xfilename,xdata3)
        
    try:
        with open(yfilename,'rb') as yfile:
            ydata3 = np.concatenate( 
               (np.load(yfile),ydata2) )
    except FileNotFoundError:
        ydata3 = ydata2
    finally:
        np.save(yfilename,ydata3)      
        
    SAVE_LOCK.release()
  
    
def load_data():
    global NP_X_DATA
    global NP_Y_DATA
    global NP_X_FIELD_DATA
    global NP_Y_FIELD_DATA
    NP_X_DATA = np.load(X_DATA_FILENAME)
    NP_Y_DATA = np.load(Y_DATA_FILENAME)
    NP_X_FIELD_DATA = np.load(X_FIELD_DATA_FILENAME)
    NP_Y_FIELD_DATA = np.load(Y_FIELD_DATA_FILENAME)


def generate_data(thread_func=run_until_done,thread_args=None):
    threads = [Thread(target=thread_func,args=thread_args) for x in range(16)]
    for i in range(16):
        threads[i].start()
    for j in range(60):
        for i in range(16):
            if not threads[i].is_alive():
                threads[i] = Thread(target=thread_func,args=thread_args)
                threads[i].start()
        sleep(1)
    print("Data collection complete, waiting for threads...")
    for i in range(16):
        threads[i].join()
        
    print("Reloading data.")


def official_run(input_model,best_only=False,render=False):
    print("DOING OFFICIAL RUN!")
    env = gsmb.make('smb-better-v3',disable_env_checker=True)
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    X_DATA = []
    Y_DATA = []
    done = False
    reward_sum = 0
    state = env.reset()
    action_plan = [ 0, 0 ]

    state, reward, done, info = env.step(action_plan[0])
    X_DATA.append([info,0,1])
    reward_sum = reward
    really_done = False
    while not done:
        if action_plan[1] == 0 or done:
            cr = 0
            cd = 0
            if X_DATA[-1][0]['life'] > info['life']:
            # The last few decisions led to a death.
                Y_DATA.append(DEATH_REWARD)
                dr = DEATH_REWIND_FRAMES
                i = 1
                while dr > 0:
                    try:
                        Y_DATA[-i] = DEATH_REWARD
                        dr -= X_DATA[-i][5]
                    except IndexError:
                        dr = 0
            else:
                cr = status_to_int(info['status']) - status_to_int(X_DATA[-1][0]['status'])
                cd = status_to_int(info['coins']) - status_to_int(X_DATA[-1][0]['coins'])
              
                Y_DATA.append(reward_sum + ( COIN_REWARD * cd) +
                                      (SIZE_REWARD * cr) )
            testdata = []
            for commit in range(1,31):
                for action in range(0,12):
                    if reward_sum == 0 and action_plan[0] == action:
                        continue
                    testdata.append([
                        info['x_pos'],
                        info['y_pos'],
                        info['world'],
                        info['stage'],
                        action,
                        commit
                        ])
            
            testdata = np.array(testdata)
            predictions = input_model.predict(testdata)
            best = np.argsort(predictions,axis=0)
            offset = -1 if best_only else randint(-30,-1)
            action_plan[0] = testdata[best[offset]][0][4]
            action_plan[1] = testdata[best[offset]][0][5]
            
                
            # print(str(COMPLEX_MOVEMENT[action_plan[0]]),"for",action_plan[1],"frames")
            X_DATA.append([info,action_plan[0],action_plan[1]])
            reward_sum = 0
        state, reward, done, info = env.step(action_plan[0])
        reward_sum += reward
        action_plan[1] -= 1
        if render:
            env.render()
    Y_DATA
    print("Done with info: ",str(info))
    if len(X_DATA) > len(Y_DATA):
        X_DATA.pop()
    env.close()
    save(X_DATA,Y_DATA,X_FIELD_DATA_FILENAME,Y_FIELD_DATA_FILENAME)
    if info['life'] == 255:
        print("Died of stupidity. Back to the drawing board...")
        return False, convert(X_DATA,Y_DATA)
    print("Actually finished!")
    return True, convert(X_DATA,Y_DATA)


def do_other_bullshit():
    env = gsmb.make('smb-better-v3',disable_env_checker=True)
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    model = tf.keras.models.Sequential( [
        tf.keras.layers.Dense(32, input_shape=(6,), activation='linear'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
        ]
        )
    loss_fn = tf.keras.losses.MeanAbsoluteError()
    model.compile(optimizer='adam', loss = loss_fn)
    model.fit(NP_X_DATA,NP_Y_DATA,epochs=5)
    model.fit(NP_X_FIELD_DATA,NP_Y_FIELD_DATA,epochs=50)
       
    training = True
    last_run = True
    while training:
        try:
            last_run,(xdata,ydata) = official_run(model, best_only=last_run,render=False)
            if last_run == True:
                break
            model.fit(xdata,ydata,epochs=50)
        except KeyboardInterrupt:
            print("Finished training.")
            training = False
            

