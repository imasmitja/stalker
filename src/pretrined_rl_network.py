# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 13:31:54 2022

@author: Usuari
"""
from rl_algorithms.maddpg import MADDPG
from rl_algorithms.matd3_bc import MATD3_BC
from rl_algorithms.masac import MASAC
from rl_algorithms.masac_qmix import MASACQMIX
import torch
import numpy as np
import os
from configparser import ConfigParser

##########################################################################################################
##############################      Reinforcement Learning Path                  ##########################
###########################################################################################################  
class rl_agent(object):
    def __init__ (self,configFile):

        config_file = os.path.dirname(os.getcwd())+'/catkin_ws/src/stalker/src/pretrined_agents/'+configFile+'_config.txt'
        print ('Configuration File   =  ',config_file)
    
        config = ConfigParser()
        config.read(config_file)
        GAMMA          = config.getfloat('hyperparam','GAMMA')
        TAU            = config.getfloat('hyperparam','TAU')
        LR_ACTOR       = config.getfloat('hyperparam','LR_ACTOR')
        LR_CRITIC      = config.getfloat('hyperparam','LR_CRITIC')
        WEIGHT_DECAY   = config.getfloat('hyperparam','WEIGHT_DECAY')
        #Scenario used to train the networks
        RNN            = config.getboolean('hyperparam','RNN')
        HISTORY_LENGTH = config.getint('hyperparam','HISTORY_LENGTH')
        DNN            = config.get('hyperparam','DNN')
        ALPHA          = config.getfloat('hyperparam','ALPHA')
        AUTOMATIC_ENTROPY = config.getboolean('hyperparam','AUTOMATIC_ENTROPY')
        DIM_1          = config.getint('hyperparam','DIM_1')
        DIM_2          = config.getint('hyperparam','DIM_2')
        # number of agents per environment
        num_agents     = config.getint('hyperparam','num_agents')
        # number of landmarks (or targets) per environment
        num_landmarks  = config.getint('hyperparam','num_landmarks')
        landmark_depth = config.getfloat('hyperparam','landmark_depth')
        #Chose device
        #DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") #To run the pytorch tensors on cuda GPU
        DEVICE = 'cpu'
        
        # initialize policy and critic
        print('Initialize the Actor-Critic networks')
        if DNN == 'MADDPG':
                self.rl_network = MADDPG(num_agents = num_agents, num_landmarks = num_landmarks, landmark_depth=landmark_depth, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE, rnn = RNN, dim_1=DIM_1, dim_2=DIM_2)
        elif DNN == 'MATD3':
                self.rl_network = MATD3_BC(num_agents = num_agents, num_landmarks = num_landmarks, landmark_depth=landmark_depth, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE, rnn = RNN, dim_1=DIM_1, dim_2=DIM_2)
        elif DNN == 'MASAC':
                self.rl_network = MASAC(num_agents = num_agents, num_landmarks = num_landmarks, landmark_depth=landmark_depth, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE, rnn = RNN, alpha = ALPHA, automatic_entropy_tuning = AUTOMATIC_ENTROPY, dim_1=DIM_1, dim_2=DIM_2)
        elif DNN == 'MASACQMIX':
                self.rl_network = MASACQMIX(num_agents = num_agents, num_landmarks = num_landmarks, landmark_depth=landmark_depth, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE, rnn = RNN, alpha = ALPHA, automatic_entropy_tuning = AUTOMATIC_ENTROPY, dim_1=DIM_1, dim_2=DIM_2)
        
        
        else:
            print('ERROR UNKNOWN DNN ARCHITECTURE')
        
        # Load the pretrained network values
        common_folder = os.path.dirname(os.getcwd()) + r"/catkin_ws/src/stalker/src/pretrined_agents/"
        trained_checkpoint = common_folder + configFile + '_episode_best.pt'
        aux = torch.load(trained_checkpoint, map_location = DEVICE)
        for i in range(num_agents):  
            # load the weights from file
            if DNN == 'MADDPG':
                self.rl_network.maddpg_agent[i].actor.load_state_dict(aux[i]['actor_params'])
                self.rl_network.maddpg_agent[i].target_actor.load_state_dict(aux[i]['target_actor_params'])
                self.rl_network.maddpg_agent[i].actor_optimizer.load_state_dict(aux[i]['actor_optim_params'])
            elif DNN == 'MATD3':
                self.rl_network.matd3_bc_agent[i].actor.load_state_dict(aux[i]['actor_params'])
                self.rl_network.matd3_bc_agent[i].target_actor.load_state_dict(aux[i]['target_actor_params'])
                self.rl_network.matd3_bc_agent[i].actor_optimizer.load_state_dict(aux[i]['actor_optim_params'])
            elif DNN == 'MASAC' or DNN == 'MASACQMIX':
                if AUTOMATIC_ENTROPY:
                    self.rl_network.masac_agent[i].actor.load_state_dict(aux[i]['actor_params'])
                    self.rl_network.masac_agent[i].actor_optimizer.load_state_dict(aux[i]['actor_optim_params'])
                    self.rl_network.masac_agent[i].alpha_optimizer.load_state_dict(aux[i]['alpha_optim_params'])
                else:
                    self.rl_network.masac_agent[i].actor.load_state_dict(aux[i]['actor_params'])
                    self.rl_network.masac_agent[i].actor_optimizer.load_state_dict(aux[i]['actor_optim_params'])
            else:
                break
        
        #Initialize history buffer with 0.
        if DNN == 'MASACQMIX':
        	self.obs_size = 10
        else:
        	self.obs_size = 7
        parallel_envs = 1
        self.history = [torch.tensor(np.zeros((parallel_envs,HISTORY_LENGTH,self.obs_size)),dtype=torch.float32)]
        #Initialize action history buffer with 0.
        self.history_a = [torch.tensor(np.zeros((parallel_envs,HISTORY_LENGTH,1)),dtype=torch.float32)] #the last entry is the number of actions, here is 1 (yaw)
        self.his_length = HISTORY_LENGTH
        self.his = [torch.cat((self.history[0],self.history_a[0]),dim=2)] #we don't use his in this case, so all 0. (however, it must be the same dimension as obs + act)
        self.obs = []
        return

    def next_action(self,observation):
        self.obs = [torch.tensor([observation])]	
        actions = self.rl_network.act(self.his,self.obs,noise=0.0)
        action = torch.stack(actions).detach().numpy().item(0)
        #update history buffer
        self.history   = [torch.cat((self.history[0],self.obs[0].reshape(1,1,self.obs_size)),dim=1)[:,1:self.his_length+1]]
        self.history_a = [torch.cat((self.history_a[0],torch.tensor(action,dtype=torch.float32).reshape(1,1,1)),dim=1)[:,1:self.his_length+1]]
        self.his = [torch.cat((self.history[0],self.history_a[0]),dim=2)]
        return action

	
	
	
	
