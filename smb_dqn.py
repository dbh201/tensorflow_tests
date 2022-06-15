# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 07:19:50 2022

@author: db_wi
"""

from __future__ import absolute_import, division, print_function

import base64
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import pyvirtualdisplay
import tensorflow as tf
import gym
import threading
from random import randint

from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.environments import parallel_py_environment
from tf_agents.environments.gym_wrapper import GymWrapper
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import categorical_q_network
from tf_agents.policies import py_tf_eager_policy,random_tf_policy
from tf_agents.policies.policy_saver import PolicySaver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec,from_spec
from tf_agents.utils import common
from tf_agents.system.system_multiprocessing import handle_main,enable_interactive_mode

from nes_py._image_viewer import ImageViewer

import smb_env_2 as gsmb
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from time import sleep,time

import visualiser
TIMEOUT_STEPS = 5000


initial_collect_steps = 1000  # @param {type:"integer"}
collect_steps_per_iteration =   1000 # @param {type:"integer"}
num_iterations = 1000 # @param {type:"integer"}
replay_buffer_capacity = 10000  # @param {type:"integer"}

batch_size = 1  # @param {type:"integer"}
# was 1e-4
learning_rate = 1e-8  # @param {type:"number"}
log_interval = 1000  # @param {type:"integer"}

gamma=0.99
num_atoms = 12  # @param {type:"integer"}
min_q_value = -15  # @param {type:"integer"}
max_q_value = 15  # @param {type:"integer"}
n_step_update = 5  # @param {type:"integer"}


num_eval_episodes = 1  # @param {type:"integer"}
eval_interval =   1000 # @param {type:"integer"}

def gym_render(a):
    return a.pyenv._envs[0].unwrapped.render()
def gym_close(a):
    if a.pyenv._envs[0].unwrapped.viewer is not None:
        return a.pyenv._envs[0].unwrapped.viewer.close()
    return None
def compute_avg_return(environment, policy, num_episodes=5,render=True):
  print("Computing average return...")
  total_return = 0.0
  for epnum in range(num_episodes):
    time_step = environment.reset()
    info = environment.pyenv._envs[0]._gym_env.env.env.env._get_info()
    episode_return = 0.0
    steps_taken = 0
    # life wraps around to 255 when you die without lives.
    print("Episode %i" % epnum)
    while not info['life'] == 255 and not info['flag_get']:
        if render:
            gym_render(environment)
        if episode_return <= -200.0:
            print("{} steps taken, cancelling with {} reward points.".format(steps_taken,episode_return))
            break        
        if steps_taken >= TIMEOUT_STEPS and episode_return < 0:
            print("{} steps taken, timing out with {} reward points.".format(steps_taken,episode_return))
            break
        action_step = policy.action(time_step)
        time_step = environment.step(action_step.action)
        episode_return += time_step.reward
        steps_taken += 1
        # env.env.env.env... more wrappers than an Amazon package
        info = environment.pyenv._envs[0]._gym_env.env.env.env._get_info()
    if info['flag_get']:
        print("We finished the level!")
    else:
        print("We ran out of lives.")
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]




def make_env(name='smb-better-v3'):
    return JoypadSpace(
    gsmb.make(name),
    COMPLEX_MOVEMENT
    )

def make_gym_suite(name='smb-better-v3'):
    return suite_gym.wrap_env(make_env(name))

def make_py_env(name='smb-better-v3',count=1):
    #if count == 1:
    envs = make_gym_suite(name)
    #else:
        #envs = parallel_py_environment.ParallelPyEnvironment([ make_gym_suite ]*count,False)
    return tf_py_environment.TFPyEnvironment(envs)


# def get_q_net(field):
#     return categorical_q_network.CategoricalQNetwork( 
#         field.observation_spec(),
#         from_spec(field.action_spec()),
#         preprocessing_layers=(
#             tf.keras.layers.Flatten(input_shape=field.observation_spec()[0].shape),
#             tf.keras.layers.Flatten(input_shape=field.observation_spec()[1].shape),
#             tf.keras.layers.Flatten(input_shape=field.observation_spec()[2].shape),
#             ),
#         preprocessing_combiner=tf.keras.layers.Concatenate(),
#         num_atoms=num_atoms,
#         fc_layer_params=fc_layer_params,
#         activation_fn=tf.nn.leaky_relu
#         )

def get_q_net(field):
    return categorical_q_network.CategoricalQNetwork( 
        field.observation_spec(),
        from_spec(field.action_spec()),
        num_atoms=num_atoms,
        fc_layer_params=fc_layer_params,
        activation_fn=tf.nn.leaky_relu
        )

def get_agent(field,q_net,optimizer,global_step):
    return categorical_dqn_agent.CategoricalDqnAgent(
        field.time_step_spec(),
        field.action_spec(),
        categorical_q_network=q_net,
        optimizer=optimizer,
        min_q_value=min_q_value,
        max_q_value=max_q_value,
        n_step_update=n_step_update,
        td_errors_loss_fn=common.element_wise_squared_loss,
        gamma=gamma,
        train_step_counter=global_step)


def load_checkpoint(agent,global_step,suffix='v3',replay_buffer=None):
    
    ckpt_dir = "DQNAgent_"+suffix
    print("Loading %s" % ckpt_dir)
    if suffix.startswith("full"):
        full_checkpoint = common.Checkpointer(
            ckpt_dir=ckpt_dir,
            max_to_keep=5,
            agent=agent,
            global_step=global_step,
            replay_buffer=replay_buffer
            )
    else:
        full_checkpoint = common.Checkpointer(
            ckpt_dir=ckpt_dir,
            max_to_keep=5,
            agent=agent,
            global_step=global_step
            )
    return full_checkpoint


def collect_step(environment, policy, replay_buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    # Add trajectory to the replay buffer
    replay_buffer.add_batch(traj)
    
def collect_random_data(agent,replay_buffer,tf_gym):
    random_policy = random_tf_policy.RandomTFPolicy(tf_gym.time_step_spec(),
                                                    tf_gym.action_spec())
    print("Collecting random data: ",end="")
    t = time()
    for _ in range(initial_collect_steps):
        collect_step(tf_gym, random_policy,replay_buffer)
    t = time() - t
    print("%.2f steps/sec" % (initial_collect_steps/t))
            
def collect_q_data(policy,replay_buffer,tf_gym):
    world = randint(1,8)
    level = randint(1,4)
    tf_gym = make_py_env('smb-better-{}-{}-v3'.format(world,level))
    print("Collecting Q data for %i-%i: " % (world,level),end="")
    t = time()
    for j in range(collect_steps_per_iteration):
        collect_step(tf_gym, policy, replay_buffer)
    t = time() - t
    print("%.2f steps/sec" % (collect_steps_per_iteration/t))
        
def train_agent(agent,replay_buffer,iterator):
    print("Training on replay data: ",end="")
    t = time()
    accs = np.empty( (num_iterations,),dtype='float')
    for i in range(num_iterations):
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss
        accs[i] = train_loss
        step = agent.train_step_counter.numpy()
    t = time() - t
    print("%.2f steps/sec" % (num_iterations/t))
    return accs
    
def focus_on_failures(agent,replay_buffer,iterator):
    print("Training with failure focus...")
    if len(loss_data) > 0:
        avg_loss = np.average(loss_data[-1])+(1.5*np.std(loss_data[-1]))
    else:
        avg_loss = 10000
    accs = np.empty( (num_iterations,),dtype='float')
    total = 0
    t = time()
    start = time()
    for i in range(num_iterations):
        exp, _ = next(iterator)
        loss = agent.train(exp).loss.numpy()
        step = agent.train_step_counter.numpy()
        total += 1
        j = 1
        if (step) % log_interval == 0:
            t = time() - t
            print('step = {0} [{2}:{3}]: {1:.2f} steps/sec'.format(step, total/t, i,j))
            t = time()
        if loss < avg_loss:
            accs[i] = loss
            continue
        while j < 10 and loss > 1:
            loss = agent.train(exp).loss.numpy()
            step = agent.train_step_counter.numpy()
            j += 1
            total += 1
            if (step) % log_interval == 0:
                t = time() - t
                print('step = {0} [{2}:{3}]: {1:.2f} steps/sec'.format(step, total/t, i,j))
                t = time()
        accs[i] = loss
    start = time() - start
    print("Rate: %.2f steps/sec" % (total/start))
    return accs

loss_data = []
avgs = []     
# parallelism is weird        
class PlotThread(threading.Thread):
    def run(self):
        self.data = None
        self.avgs = None
        while self.data is None or self.avgs is None:
            sleep(1)
            
        plt.subplot(2,1,1)
        plt.ylabel('count')
        p=10
        a = max([np.average(d) for d in loss_data])
        stdev = max([np.std(d) for d in loss_data])
        
        t = a + 1.5*stdev
        while p < t:
            p *= 10
        p //= 10
        m = p
        while m < t:
            m += p
        plt.hist([ np.clip(x,0,m) for x in loss_data],range=(0,m))
        plt.subplot(2,1,2)
        plt.ylabel('average')
        for a in range(len(avgs)):
            plt.scatter( a, avgs[a] )
        plt.plot([np.average(x) for x in loss_data],'-c')
        plt.show()
            
    

def main(argv=None):
    global loss_data
    global fc_layer_params
    global avgs
    global agent
    global iterator
    
    if argv:
        suffix = argv
    else:
        suffix = input("Input DQNAgent name or press enter for default:").strip() or "1024_512_256_128_v3_ram"
    fc_layer_params=[]
    for token in suffix.split('_'):
        t = None
        try:
            t = int(token)
        except ValueError:
            pass
        if t == None:
            if len(fc_layer_params) > 0:
                break
        else:
            fc_layer_params.append(t)
    
    

    fc_layer_params=tuple(fc_layer_params)
    print(fc_layer_params)
    print("Generating environments...")  
    field = make_py_env('smb-better-vanilla-v3')
    train = make_py_env('smb-better-{}-{}-v3'.format(randint(1,8),randint(1,4)))
    
    
    def show_loss_graph(join=False):
        pt = PlotThread()
        pt.start()
        pt.suffix = suffix
        pt.data = loss_data
        pt.avgs = avgs
        if join:
            pt.join()
        
    def auto_train():
        global loss_data
        print("Auto training...")
        for i in range(repeat):
            print("%i of %i" % (i+1,repeat))
            collect_q_data(agent.collect_policy,replay_buffer,train)
            a = train_agent(agent,replay_buffer,iterator)
        print("Setting results...")
        if len(loss_data) > 3:
            loss_data.pop(0)
        loss_data.append(a)
        b = compute_avg_return(field,agent.policy,1)
        if len(avgs) > 3:
            avgs.pop(0)
        avgs.append(b)
        show_loss_graph(True)
        
            
        
    print("Initializing Q-net...")
    q_net = get_q_net(field)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    
    global_step = tf.Variable(0)
    print("Initializing agent...")
    agent = get_agent(field,q_net,optimizer,global_step)
    
    
    agent.initialize()
    agent.train = common.function(agent.train)
    agent.train_step_counter.assign(0)
    
    #@test {"skip": true}
    print("Initializing replay buffer...")
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=batch_size,
        max_length=replay_buffer_capacity)
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, sample_batch_size=batch_size,
        num_steps=n_step_update + 1).prefetch(3)
    iterator = iter(dataset)
    
    official_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=batch_size,
        max_length=replay_buffer_capacity)
    
    official_dataset = official_buffer.as_dataset(
        num_parallel_calls=3, sample_batch_size=batch_size,
        num_steps=n_step_update + 1).prefetch(3)
    
    official_iterator = iter(official_dataset)
    chkpt = load_checkpoint(agent,global_step,suffix,None)
    select = 0
    repeat = 1
    save_required = False
    options = [
        ("collect_random_data",collect_random_data, (agent,replay_buffer,train),False),
        ("collect_q_data",collect_q_data, (agent.collect_policy,replay_buffer,train),False),
        ("collect_qoff_data",collect_q_data, (agent.policy,official_buffer,train),False),
        ("train_agent",train_agent,(agent,replay_buffer,iterator),True),
        ("train_agent_qoff",train_agent,(agent,official_buffer,official_iterator),True),
        ("auto_train",auto_train,(),False),
        ("focus_on_failures",focus_on_failures,(agent,replay_buffer,iterator),True),
        ("focus_qoff_failures",focus_on_failures,(agent,official_buffer,official_iterator),True),
        ("compute_avg_return",compute_avg_return,(field,agent.policy,1),False),
        ("show_loss_graph",show_loss_graph,(),False),
        ("set_repeat",None,None,False),
        ("save",lambda a: a,(),True),
        ("exec",eval,None,False),
        ("quit",None,None,True)
        ]
    
    if global_step == 0:
        print("New system found. Initializing with random data...")
        loss_data = []
        avgs = []
        options[0][1](*options[0][2])
        loss_data.append(options[3][1](*options[3][2]))
        avgs.append(compute_avg_return(field,agent.policy,1,False))
        
    while True:
        try:
            print("DQNAgent {} ready at step {}".format(suffix,global_step.numpy()))
            for a in range(len(options)):
                print("{}: {}".format(a,options[a][0]))
            print("Will repeat {} times".format(repeat))
            
            workles = input("Input function number: ")
            select = int(workles)
            selection = options[select]
        except ValueError:
            print("Invalid input.")
            continue
        except IndexError:
            print("Invalid selection!")
            continue
        except KeyboardInterrupt:
            print("Quitting...")
            break
        
        try:
            train = make_py_env('smb-better-{}-{}-v3'.format(randint(1,8),randint(1,4)))
            save_required = selection[-1]
            if selection[0] == 'set_repeat':
                try:
                    repeat = int(input("Input repeat count: "))
                except ValueError:
                    print("Invalid input.")
            elif selection[0] == 'quit':
                print("Quitting...")
                break
            elif selection[0] == 'exec':
                print("Input expression to exec below:")
                try:
                    exec(input(),globals(),locals())
                except Exception as e:
                    print(type(e),':',e)
            elif selection[0] == 'show_loss_graph':
                selection[1](*selection[2])
            elif selection[0] == 'auto_train':
                selection[1]()
            elif selection[0] == 'compute_avg_return':
                if len(avgs) > 3:
                    avgs.pop(0)
                avgs.append(selection[1](*selection[2]))
            else:
                for rep in range(repeat):
                    print("Rep %i of %i" % (rep + 1,repeat))
                    a = selection[1](*selection[2])
                if a is not None:
                    if len(loss_data) > 3:
                        loss_data.pop(0)
                    loss_data.append(a)
                
        except KeyboardInterrupt:
            print("Cancelling...")
        finally:
            if save_required:
                print("Saving checkpoints...")
                chkpt.save(agent.train_step_counter)
            gym_close(train)
            gym_close(field)

main("256_v3_ram")    
#if(global_step < 1000):
#    collect_random_data(agent,replay_buffer,train,iterator)
#train_agent(agent,replay_buffer,iterator,train,field,chkpt)