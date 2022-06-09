#!/usr/bin/env python
# license removed for brevity
# -*- coding: utf-8 -*-
"""
Created on March 029 2020

@author: Ivan Masmitja Rusinol

Project: AIforUTracking
"""

import numpy as np
import random
import time
import socket
import rospy
import serial
import sys

SOUND_SPEED = 1500.

#############################################################
## Particle Filter
############################################################
#For modeling the target we will use the TargetClass with the following attributes 
#and functions:
class ParticleFilter(object):
    """ Class for the Particle Filter """
 
    def __init__(self,std_range,init_velocity,dimx,particle_number = 6000, method = 'range', max_pf_range = 250):
 
        self.std_range = std_range
        self.init_velocity = init_velocity 
        self.x = np.zeros([particle_number,dimx])
        self.oldx = np.zeros([particle_number,dimx])
        self.particle_number = particle_number
        
        self._x = np.zeros([dimx])
       
        # target's noise
        self.forward_noise = 0.0
        self.turn_noise = 0.0
        self.sense_noise = 0.0
        self.velocity_noise = 0.0
        
        # time interval
        self.dimx=dimx
        
        self._velocity = 0
        self._orientation = 0
        
        #Weights
        self.w = np.ones(particle_number)
        
        #Covariance of the result
        self.covariance_vals = [0.,0.]
        self.covariance_theta = 0.
        
        #Flag to initialize the particles
        self.initialized = False
        
        #save actual data as a old to be used on TDOA method
        self.measurement_old = 0
        self.dist_all_old = np.zeros(particle_number)
        self.w_old = self.w
        self.observer_old = np.array([0,0,0,0])
        
        self.method = method
        #covariance matrix of final estimation
        self.cov_matrix = np.ones([2,2])
        
        #maximum target range
        self.max_pf_range = max_pf_range
        
        
    def target_estimation(self):
        """ Calculate the mean error of the system
        :param r: current target object
        :param p: particle set
        :return mean error of the system
        """
        #8- Target prediction (we predict the best estimation for target's position = mean of all particles)
        sumx = 0.0
        sumy = 0.0
        sumvx = 0.0
        sumvy = 0.0

        method = 2
        if method == 1:
            for i in range(self.particle_number):
               sumx += self.x[i][0]
               sumy += self.x[i][2]
               sumvx += self.x[i][1]
               sumvy += self.x[i][3]
            self._x = np.array([sumx, sumvx, sumy, sumvy])/self.particle_number
            self._velocity = np.sqrt(self._x[1]**2+self._x[3]**2)
            self._orientation = np.arctan2(self._x[3],self._x[1])
        if method == 2:
            for i in range(self.particle_number):
               sumx += self.x[i][0]*self.w[i]
               sumy += self.x[i][2]*self.w[i]
               sumvx += self.x[i][1]*self.w[i]
               sumvy += self.x[i][3]*self.w[i]
            self._x = np.array([sumx, sumvx, sumy, sumvy])/np.sum(self.w)
            self._velocity = np.sqrt(self._x[1]**2+self._x[3]**2)
            self._orientation = np.arctan2(self._x[3],self._x[1])
        #finally the covariance matrix is computed. 
        #http://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix/
        xarray = self.x.T[0]
        yarray = self.x.T[2]
        self.cov_matrix = np.cov(xarray, yarray)
        return

    def init_particles(self,position,slantrange):
    	
        for i in range(self.particle_number):
            #Random distribution with circle shape
            t = 2*np.pi*np.random.rand()
            if self.method == 'area':
                r = np.random.rand()*self.max_pf_range*2 - self.max_pf_range
            else:
                r = np.random.rand()*self.std_range*2 - self.std_range + slantrange
            
            self.x[i][0] = r*np.cos(t)+position[0]
            self.x[i][2] = r*np.sin(t)+position[2]
            #target's orientation
            orientation = np.random.rand() * 2.0 * np.pi   # target's orientation
            # target's velocity 
            v = random.gauss(self.init_velocity, self.init_velocity/2)  
            self.x[i][1] = np.cos(orientation)*v
            self.x[i][3] = np.sin(orientation)*v
        self.target_estimation()
        self.initialized = True
        print('WARNING: Particles initialized')
        return
    
    #Noise parameters can be set by:
    def set_noise(self, forward_noise, turn_noise, sense_noise, velocity_noise):
        """ Set the noise parameters, changing them is often useful in particle filters
        :param new_forward_noise: new noise value for the forward movement
        :param new_turn_noise:    new noise value for the turn
        :param new_sense_noise:  new noise value for the sensing
        """
        # target's noise
        self.forward_noise = forward_noise
        self.turn_noise = turn_noise
        self.sense_noise = sense_noise
        self.velocity_noise = velocity_noise

    #Move particles acording to its motion
    def predict(self,dt):
        """ Perform target's turn and move
        :param turn:    turn command
        :param forward: forward command
        :return target's state after the move
        """
        gaussnoise = False
        for i in range(self.particle_number):
            # turn, and add randomness to the turning command
            turn = np.arctan2(self.x[i][3],self.x[i][1])
            if gaussnoise == True:
                orientation = turn + random.gauss(0.0, self.turn_noise)
            else:
                orientation = turn +  np.random.rand()*self.turn_noise*2 -self.turn_noise
            orientation %= 2 * np.pi
         
            # move, and add randomness to the motion command
            velocity = np.sqrt(self.x[i][1]**2+self.x[i][3]**2)
            forward = velocity*dt
            if gaussnoise == True:
                dist = float(forward) + random.gauss(0.0, self.forward_noise)
            else:
                dist = float(forward) + np.random.rand()*self.forward_noise*2 - self.forward_noise
            self.x[i][0] = self.x[i][0] + (np.cos(orientation) * dist)
            self.x[i][2] = self.x[i][2] + (np.sin(orientation) * dist)
            if gaussnoise == True:
                newvelocity = velocity + random.gauss(0.0, self.velocity_noise)
            else:
                newvelocity = velocity + np.random.rand()*self.velocity_noise*2 - self.velocity_noise
            if newvelocity < 0:
                newvelocity = 0
            self.x[i][1] = np.cos(orientation) * newvelocity
            self.x[i][3] = np.sin(orientation) * newvelocity
        return 

    #To calculate Gaussian probability:
    @staticmethod
    def gaussian(self,mu_old,mu, sigma, z_old,z,inc_observer):
        """ calculates the probability of x for 1-dim Gaussian with mean mu and var. sigma
        :param mu:    distance to the landmark
        :param sigma: standard deviation
        :param x:     distance to the landmark measured by the target
        :return gaussian value
        """
        if self.method == 'area':
            sigma = 1. #was 5
            particlesRange = self.max_pf_range 
            # calculates the probability of x for 1-dim Gaussian with mean mu and var. sigma in a filled circle shape
            # We use the Cauchy distribution (https://en.wikipedia.org/wiki/Cauchy_distribution)
            if z != -1: #a new ping is received -> #all particles outside the tagrange have a small weight; #all particles inside the tagrange have a big weight
                return (1/2.)-(1/np.pi)*np.arctan((mu-particlesRange)/sigma)
            else: #no new ping is received -> #all particles outside the tagrange have a big weight; #all particles inside the tagrange have a small weight
                sigma = 40.
                return (1/2.)+(1/np.pi)*np.arctan((mu-particlesRange)/sigma)
        else:
            # calculates the probability of x for 1-dim Gaussian with mean mu and var. sigma
            return np.exp(- ((mu - z) ** 2) / (sigma ** 2) / 2.0) / np.sqrt(2.0 * np.pi * (sigma ** 2))
    
    #The next function we will need to assign a weight to each particle according to 
    #the current measurement. See the text below for more details. It uses effectively a 
    #Gaussian that measures how far away the predicted measurements would be from the 
    #actual measurements. Note that for this function you should take care of measurement 
    #noise to prevent division by zero. Such checks are skipped here to keep the code 
    #as short and compact as possible.
    def measurement_prob(self, measurement,observer):
        """ Calculate the measurement probability: how likely a measurement should be
        :param measurement: current measurement
        :return probability
        """
        #The closer a particle to a correct position, the more likely will be the set of 
            #measurements given this position. The mismatch of the actual measurement and the 
            #predicted measurement leads to a so-called importance weight. It tells us how important 
            #that specific particle is. The larger the weight, the more important it is. According 
            #to this each of our particles in the list will have a different weight depending on 
            #a specific target measurement. Some will look very plausible, others might look 
            #very implausible.           
        dist_all = []
        equal = 0
        for i in range(self.particle_number):
            dist = np.sqrt((self.x[i][0] - observer[0])**2 + (self.x[i][2] - observer[2])**2)
            dist_old = np.sqrt((self.x[i][0] - self.observer_old[0])**2 + (self.x[i][2] - self.observer_old[2])**2)
            inc_observer = np.sqrt((observer[0] - self.observer_old[0])**2 + (observer[2] - self.observer_old[2])**2)
            self.w[i] = self.gaussian(self,dist_old,dist, self.sense_noise, self.measurement_old,measurement,inc_observer)
            inc_mu = (self.dist_all_old[i]-dist)
            inc_z = (self.measurement_old-measurement)
            if (inc_mu >= 0 and inc_z >= 0) or (inc_mu < 0 and inc_z < 0):
                equal +=1
            dist_all.append(dist)
            
        #save actual data as a old to be used on TDOA method
        self.measurement_old = measurement
        self.dist_all_old = np.array(dist_all)
        self.w_old=self.w
        self.observer_old = observer
        return 
    
    def resampling(self,z):
        #After that we let these particles survive randomly, but the probability of survival 
            #will be proportional to the weights.
            #The final step of the particle filter algorithm consists in sampling particles from 
            #the list p with a probability which is proportional to its corresponding w value. 
            #Particles in p having a large weight in w should be drawn more frequently than the 
            #ones with a small value
            #Here is a pseudo-code of the resampling step:
            #while w[index] < beta:
            #    beta = beta - w[index]
            #    index = index + 1
            #    select p[index]
                        
        method = 2 #NO compound method
        #method = 3.2 #compound method
        #method = 3 #compound method presented in OCEANS'18 Kobe
        
        if method == 1:   
            # 4- resampling with a sample probability proportional
            # to the importance weight
            p3 = np.zeros([self.particle_number,self.dimx])
            index = int(np.random.random() * self.particle_number)
            beta = 0.0
            mw = max(self.w)
            for i in range(self.particle_number):
                beta += np.random.random() * 2.0 * mw
                while beta > self.w[index]:
                    beta -= self.w[index]
                    index = (index + 1) % self.particle_number
                p3[i]=self.x[index]
            self.x = p3
            return
        if method == 2:
            #From: https://classroom.udacity.com/courses/ud810/lessons/3353208568/concepts/33538586070923
            # Systematic Resampling
            p3 = np.zeros([self.particle_number,self.dimx])
            ci = np.zeros(self.particle_number)
            normalized_w = self.w/np.sum(self.w)
            ci[0]=normalized_w[0]
            for i in range(1,self.particle_number):
                ci[i]=ci[i-1]+normalized_w[i]
            u = np.random.random()/self.particle_number
            i = 0
            for j in range(self.particle_number):
                while (u > ci[i]):
                    i += 1
                p3[j]=self.x[i]
                u = u + 1./self.particle_number
            self.x = p3
            return
        if method == 3: #this mehtod works ok and was presented in OCEANS Kobe 2018
            # Systematic Resampling + random resampling
            if self.particle_number == 10000:
                ratio = 640 #160 works ok; ratio=10 is ok for statik targets
            elif self.particle_number == 6000:
                ratio = 400 #100 works ok; ratio=10 is ok for statik targets
            elif self.particle_number == 3000:
                ratio = 200 #50 works ok; ratio=10 is ok for statik targets
            elif self.particle_number == 1000:
                ratio = 120 #15 works ok; ratio=10 is ok for statik targets
            else:
                ratio = 50 #50 works ok; ratio=10 is ok for statik targets
            radii = 5 #50 works ok
            #From: https://classroom.udacity.com/courses/ud810/lessons/3353208568/concepts/33538586070923
            p3 = np.zeros([self.particle_number,self.dimx])
            ci = np.zeros(self.particle_number)
            normalized_w = self.w/np.sum(self.w)
            ci[0]=normalized_w[0]
            for i in range(1,self.particle_number):
                ci[i]=ci[i-1]+normalized_w[i]
            u = random.random()/(self.particle_number-ratio)
            i = 0
            for j in range((self.particle_number-ratio)):
                while (u > ci[i]):
                    i += 1
                p3[j]=self.x[i]
                u = u + 1./(self.particle_number-ratio)
                
            for i in range(ratio):
                #Random distribution with circle shape
                aux=np.zeros(4)
                t = 2*np.pi*np.random.rand()
                r = np.random.rand()*radii
                aux[0] = r*np.cos(t)+self._x[0]
                aux[2] = r*np.sin(t)+self._x[2]
                #target's orientation
                orientation = np.random.rand() * 2.0 * np.pi   # target's orientation
                # target's velocity 
                v = random.gauss(self.init_velocity, self.init_velocity/2.)  
                aux[1] = np.cos(orientation)*v
                aux[3] = np.sin(orientation)*v
                p3[j+i+1]= aux
                self.w[j+i+1] = 1./(self.particle_number/3.)
            self.x = p3
            return
        if method == 3.2: 
            #this mehtod is a modification used in TAG-Only tracking, is similar than the method presented in OCEANS Kobe 2018
            #the main difference is that the random resampling is centred over the WG position instead of the Target estimation
            # Systematic Resampling + random resampling
            ratio = 50 #50 works ok
            radii = self.max_pf_range #50 works ok
            
            #From: https://classroom.udacity.com/courses/ud810/lessons/3353208568/concepts/33538586070923
            p3 = np.zeros([self.particle_number,self.dimx])
            ci = np.zeros(self.particle_number)
            normalized_w = self.w/np.sum(self.w)
            ci[0]=normalized_w[0]
            for i in range(1,self.particle_number):
                ci[i]=ci[i-1]+normalized_w[i]
            u = np.random.random()/(self.particle_number-ratio)
            i = 0
            for j in range((self.particle_number-ratio)):
                while (u > ci[i]):
                    i += 1
                p3[j]=self.x[i]
                u = u + 1./(self.particle_number-ratio)
                
            for i in range(ratio):
                i += 1
                #Random distribution with circle shape
                aux=np.zeros(4)
                t = 2*np.pi*np.random.rand()
                r = np.random.rand()*radii
                aux[0] = r*np.cos(t)+self.observer_old[0]
                aux[2] = r*np.sin(t)+self.observer_old[2]
                #target's orientation
                orientation = np.random.rand() * 2.0 * np.pi   # target's orientation
                # target's velocity 
                v = random.gauss(self.init_velocity, self.init_velocity/2.)  
                aux[1] = np.cos(orientation)*v
                aux[3] = np.sin(orientation)*v
                p3[j+i]= aux
                self.w[j+i] = 1/10000.
            self.x = p3
            return
    
    
    #6- It computes the average error of each particle relative to the target pose. We call 
            #this function at the end of each iteration:
            # here we get a set of co-located particles   
    #At every iteration we want to see the overall quality of the solution, for this 
    #we will use the following function:
    def evaluation(self,observer,z,max_error=50):
        """ Calculate the mean error of the system
        :param r: current target object
        :param p: particle set
        :return mean error of the system
        """
        if self.method != 'area':
        
            #Evaluate the distance error
            sum2 = 0.0
            for i in range(self.particle_number):
                # Calculate the mean error of the system between Landmark (WG) and particle set
                dx = (self.x[i][0] - observer[0])
                dy = (self.x[i][2] - observer[2])
                err = np.sqrt(dx**2 + dy**2)
                sum2 += err
            print('Evaluation -> distance error: ',abs(sum2/self.particle_number - z))
            
            #Evaluate the covariance matrix
            err_x = self.x.T[0]-self._x[0]
            err_y = self.x.T[2]-self._x[2]
            cov = np.cov(err_x,err_y)
            # Compute eigenvalues and associated eigenvectors
            vals, vecs = np.linalg.eig(cov)
            confidence_int = 2.326**2
            self.covariance_vals = np.sqrt(vals) * confidence_int
            # Compute tilt of ellipse using first eigenvector
            vec_x, vec_y = vecs[:,0]
            self.covariance_theta = np.arctan2(vec_y,vec_x)
            print('Evaluation -> covariance (CI of 98): %.2f m(x) %.2f m(y) %.2f deg'%(self.covariance_vals[0],self.covariance_vals[1],np.degrees(self.covariance_theta)))
            
            #if abs(sum2/self.particle_number - z) > max_error:
            #	self.initialized = False
            if abs(sum2/self.particle_number - z) > max_error or np.sqrt(self.covariance_vals[0]**2+self.covariance_vals[1]**2) < 0.1:
                self.initialized = False
                self.init_particles(position=observer, slantrange=z)
        else:
            if np.max(self.w) < 0.1:
                self.initialized = False
            #Compute maximum particle dispersion:
            max_dispersion = np.sqrt((np.max(self.x.T[0])-np.min(self.x.T[0]))**2+(np.max(self.x.T[2])-np.min(self.x.T[2]))**2)     
        return 


##########################################################################################################
##############################                    TARGET CLASS   ##########################################
###########################################################################################################
class Target(object):
    
    def __init__(self,method='range',max_pf_range=250):
        #Target parameters
        self.method = method
        
        #PF initialization #######################################################################################
        self.pxs=[]
        #Our particle filter will maintain a set of n random guesses (particles) where 
        #the target might be. Each guess (or particle) is a vector containing [x,vx,y,vy]
        # create a set of particles
        # sense_noise is not used in area-only
        
        #self.pf = ParticleFilter(std_range=10.,init_velocity=.1,dimx=4,particle_number=10000,method=method,max_pf_range=max_pf_range)
        #self.pf.set_noise(forward_noise = 0.01, turn_noise = 0.01, sense_noise=5., velocity_noise = 0.01)
        
        #to do tests in the CIRS test tank?
        #self.pf = ParticleFilter(std_range=1.,init_velocity=.05,dimx=4,particle_number=10000,method=method,max_pf_range=max_pf_range)
        #self.pf.set_noise(forward_noise = 0.001, turn_noise = 0.01, sense_noise=2., velocity_noise = 0.001)
        
        #as BSC RL tests in Python
        #self.pf = ParticleFilter(std_range=20.,init_velocity=.2,dimx=4,particle_number=10000,method=method,max_pf_range=max_pf_range)
        #self.pf.set_noise(forward_noise = 0.01, turn_noise = 0.1, sense_noise=5., velocity_noise = 0.01)
        
        self.pf = ParticleFilter(std_range=20.,init_velocity=0.4,dimx=4,particle_number=10000,method=method,max_pf_range=max_pf_range)
        # self.pf.set_noise(forward_noise = 0.01, turn_noise = .5, sense_noise=2., velocity_noise = 0.01)
        self.pf.set_noise(forward_noise = 1., turn_noise = .9, sense_noise=5., velocity_noise = 0.01)
        
        self.position = [0.,0.,0.,0.]
        
        #############LS initialization###########################################################################
        self.lsxs=[]
        self.eastingpoints_LS=[]
        self.northingpoints_LS=[]
        self.Plsu=np.array([])
        self.allz=[]

    
    #############################################################################################
    ####             Particle Filter (PF)                                                      ##         
    #############################################################################################                               
    def updatePF(self,dt,new_range,z,myobserver,update=True):
        max_error = 150. #at test tank maybe put it at 10
        init_time = rospy.get_time()
        if update == True:
                  
            # Initialize the particles if needed
            if self.pf.initialized == False:
                self.pf.init_particles(position=myobserver, slantrange=z)
                
            #we save the current particle positions to plot as the old ones
            self.pf.oldx = self.pf.x+0. 
            
            # Predict step (move all particles)
            self.pf.predict(dt)
            
            # Update step (weight and resample)
            if new_range == True:     
                # Update the weiths according its probability
                self.pf.measurement_prob(measurement=z,observer=myobserver)
                if max(self.pf.w) == 0:
                    self.pf.init_particles(position=self.pf.previous_observer, slantrange=self.pf.previous_z)
                    self.pf.measurement_prob(measurement=z,observer=myobserver,error_mult=50.)       
                #Resampling        
                self.pf.resampling(z)
                # Calculate the avarage error. If it's too big the particle filter is initialized                    
                self.pf.evaluation(observer=myobserver,z=z,max_error=max_error)    
            # We compute the average of all particles to fint the target
            self.pf.target_estimation()
        #Save position
        self.position = self.pf._x
        pf_time=rospy.get_time()-init_time
        return pf_time
        
    #############################################################################################
    ####             Least Squares Algorithm  (LS)                                             ##         
    #############################################################################################
    def updateLS(self,dt,new_range,z,myobserver):
        num_ls_points_used = 30
        init_time = rospy.get_time()
        #Propagate current target state estimate
        if new_range == True:
            self.allz.append(z)
            self.eastingpoints_LS.append(myobserver[0])
            self.northingpoints_LS.append(myobserver[2])
        numpoints = len(self.eastingpoints_LS)
        if numpoints > 3:
            #Unconstrained Least Squares (LS-U) algorithm 2D
            #/P_LS-U = N0* = N(A^T A)^-1 A^T b
            #where:
            P=np.matrix([self.eastingpoints_LS[-num_ls_points_used:],self.northingpoints_LS[-num_ls_points_used:]])
            # N is:
            N = np.concatenate((np.identity(2),np.matrix([np.zeros(2)]).T),axis=1)
            # A is:
            num = len(self.eastingpoints_LS[-num_ls_points_used:])
            A = np.concatenate((2*P.T,np.matrix([np.zeros(num)]).T-1),axis=1)
            # b is:
            b = np.matrix([np.diag(P.T*P)-np.array(self.allz[-num_ls_points_used:])*np.array(self.allz[-num_ls_points_used:])]).T
            # Then using the formula "/P_LS-U" the position of the target is:
            try:
                self.Plsu = N*(A.T*A).I*A.T*b
            except:
                print('WARNING: LS singular matrix')
                try:
                    self.Plsu = N*(A.T*A+1e-6).I*A.T*b
                except:
                    pass

        #Compute MAP orientation and save position
        try:
            ls_velocity = np.array([(self.Plsu[0]-self.lsxs[-1][0])/dt,(self.Plsu[1]-self.lsxs[-1][1])/dt])
        except IndexError:
            ls_velocity = np.array([0,0])
        try:
            ls_position = np.array([self.Plsu.item(0),ls_velocity.item(0),self.Plsu.item(1),ls_velocity.item(1)])
        except IndexError:
            #ls_position = np.array([myobserver[0],ls_velocity[0],myobserver[2],ls_velocity[1]])
            ls_position = np.array([0.,ls_velocity[0],0.,ls_velocity[1]])
        self.lsxs.append(ls_position)
        #Save position
        self.position = ls_position
        ls_time=rospy.get_time()-init_time
        return ls_time

##########################################################################################################
##############################      NETCAT CLASS to conecti with EVOLOGICS       ##########################
###########################################################################################################  
#Create a netcat connection socket for each modem that we whant control
class netcat(object):
    def __init__ (self,hostname,port,modem_name, debug = False, interface = 'ethernet', serial_port_name = '', sim = False):
        if interface == 'ethernet':
        	print('Initializing Ethernet modem ('+modem_name+') at address '+str(hostname))
        elif interface == 'serial':
        	print('Initializing Serial modem ('+modem_name+') at port '+str(serial_port_name))
        else:
        	print('ERROR: Modem interface does not exist')
        self.ip = hostname
        self.port = port
        self.name = modem_name
        self.interface = interface
        self.sim = sim
        self.serial_port_name = serial_port_name
        #print for debug
        self.debug_print = debug
        self.python_version = sys.version_info[0]
        #Configuration
        if self.sim == False:
        	data = self.send(b'+++ATC')
        #self.dprint('Reset ' + modem_name)
        #data = self.port_write(b'ATZ0')
        #wait for reset completed
        #rospy.sleep(2.)
        #Configuration
        self.dprint('Start Configuration:')
        data = self.send(b'AT@ZX1')
        #Read status
        self.dprint('Read Status:')
        data = self.send(b'AT?S')
        print('Modem ('+modem_name+') initialized')
    
    def dprint(self,message):
        if self.debug_print == True:
            print(message)
        return
        
    def port_write(self,command):
        if self.python_version == 2:
        	self.dprint('Sent --> ' + self.name +':' + command)
        else:
        	self.dprint('Sent --> ' + self.name +':' + command.decode('utf-8'))
        if self.interface == 'ethernet':
        	self.netcat = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        	self.netcat.connect((self.ip, self.port))
        	self.netcat.sendall(command + b'\n')
        	self.netcat.shutdown(socket.SHUT_WR)
        	self.netcat.shutdown(socket.SHUT_RD)
        	self.netcat.close()
        elif self.interface == 'serial':
        	self.serialPort = serial.Serial(port=self.serial_port_name, baudrate=19200, timeout=5)
        	self.serialPort.write(command + b'\r')
        	self.serialPort.close()
        else:
        	print('ERROR: Modem interface does not exist')
        return
    
    def port_read(self):
        if self.interface == 'ethernet':
        	self.netcat = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        	self.netcat.connect((self.ip, self.port))
        	self.netcat.shutdown(socket.SHUT_WR)
        elif self.interface == 'serial':
        	self.serialPort = serial.Serial(port=self.serial_port_name, baudrate=19200, timeout=5)
        else:
        	print('ERROR: Modem interface does not exist')
        buff = ''
        while 1:
        	if self.interface == 'ethernet':
        		if self.python_version == 2:
        			read = self.netcat.recv(1)
        		else:
        			read = self.netcat.recv(1).decode('utf-8')
        	elif self.interface == 'serial':
        		if self.python_version == 2:
        			read = self.serialPort.read()
        		else:
        			read = self.serialPort.read().decode('utf-8')
        	else:
        		print('ERROR: Modem interface does not exist')
        	if read:
        		buff += read
        	else:
        		break
        if self.interface == 'ethernet':
        	#self.netcat.shutdown(socket.SHUT_RD)
        	self.netcat.close()
        elif self.interface == 'serial':        	
        	self.serialPort.close()
        	
        return buff

    def send(self,command):
        self.port_write(command)
        data = ''
        while 1:
            read = self.port_read()
            if read == "":
                #print 'empty'
                break
            data = data + read 
            self.dprint("Received <-- " + self.name + ':' + read[0:-2])
        return data

    def send2(self,command):
        if self.python_version == 2:
        	self.dprint('Sent --> '+ self.name +':' + command)
        else:
        	self.dprint('Sent --> '+ self.name +':' + command.decode('utf-8'))
        if self.interface == 'ethernet':
        	self.netcat = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        	self.netcat.connect((self.ip, self.port))
        	self.netcat.sendall(command + b'\n')
        	self.netcat.shutdown(socket.SHUT_WR)
        	self.netcat.shutdown(socket.SHUT_RD)
        	self.netcat.close()
        elif self.interface == 'serial':
        	self.serialPort = serial.Serial(port=self.serial_port_name, baudrate=19200, timeout=5)
        	self.serialPort.write(command + b'\r')
        	self.serialPort.close()
        else:
        	print('ERROR: Modem interface does not exist')        
        return 

    def reset(self):
        self.port_write(b'AT%RESET')
        for i in range(10):
            read = self.port_read()
            if read.find('OK') != -1:
                        return
            self.dprint("Received <-- " + self.name + ':' + read[0:-2])
        self.dprint('Error during reset')
        return

    def send_ack(self,command,max_time):
        #first we send the command
        self.send2(command)
        #then we look for the ack
        init_time = rospy.get_time()
        delivered = False
        failed_num = 0
        count = 0
        data = ''
        while 1:
            #Open port to receive response
            read = self.port_read() 
            if read == "":
                #self.dprint('empty')
                if delivered == True:
                    break
            else:
                self.dprint("Received <-- " + self.name + ':\r\n' + repr(read))
                data = data+read #save without '\r\n'
                #Check comunication status
                count += read.count('SENDEND') #increase counter for one transmision attempt
                count -= read.count('RECVEND') #discount from counter 1 for reception accepted
                if read.find('USBL') != -1 and count < 1 or read.find('DELIVERED') != -1:
                    delivered = True  
                if read.find('FAILEDIM') != -1:
                    failed_num += 1
                    if failed_num > 3:
                        self.dprint('Communication error num %d: FAILEDIM' % failed_num)
                        break
            #Check the maximum time for communication
            if rospy.get_time()-init_time > max_time:
                self.dprint('Communication error: Time Limit Exceeded')
                break  
        return data.split('\r\n'),delivered,failed_num
        
    def move(self,x,y,z):
        self.netcat = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        port2 = 11000
        self.netcat.connect((self.ip, port2))
        message = str(x)+' '+str(y)+' '+str(z) + '\n'
        if self.python_version == 2:
        	self.netcat.sendall(message)
        else:
        	self.netcat.sendall(message.encode('utf-8'))
        self.netcat.shutdown(socket.SHUT_WR)
        self.netcat.close()
        data = ''
        while 1:
            self.netcat = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.netcat.connect((self.ip, port2))
            self.netcat.shutdown(socket.SHUT_WR)
            if self.python_version == 2:
            	read = self.netcat.recv(1024)
            else:
            	read = self.netcat.recv(1024).decode('utf-8')
            #self.netcat.shutdown(socket.SHUT_RD)
            self.netcat.close()
            if read == "":
                break
            data = data+read 
        return 
    
    def slant_range(self,remot_modem_address):
        #first we send a recet to cancel any previous command or requestand clear transmit buffer
        #self.reset()
        #Send Instant (IM) messages with ack to obtain slant range measuremen
        if self.sim == False: 
        	message = 'AT*SENDIM,p0,1,'+str(remot_modem_address)+',ack,-'
        else:
        	message = 'AT*SENDIM,1,'+str(remot_modem_address)+',ack,-'
        if self.python_version == 2:
        	data, ack, failed_num = self.send_ack(message,5)
        else:
        	data, ack, failed_num = self.send_ack(message.encode('utf-8'),5)
        print('data=',data)
        print('ack=',ack)
        print('failed_num=', failed_num)
        '''       
        #Find parameters in IM notification
        start_timestamp = []
        end_timestamp = []
        #Find the start and end timestamps from the data received
        for i in range(len(data)):
            if data[i].find('SENDEND') != -1 :
                start_timestamp.append(int(data[i].split(',')[3]))
            if data[i].find('RECVEND') != -1:
                aux   = data[i].split(',')
                end_timestamp.append(int(aux[1]))
                rssi = int(aux[3])
                integrity = int(aux[4])
        #Check if we received start and end timestamp, ifnot some error occurred.
        if len(start_timestamp) == 0 or len(end_timestamp) == 0:
            self.dprint('Range error occurred')
            return -1
        #If we have obtained more than one timestamp we choose the correct one
        if len(start_timestamp)>1 and failed_num > 0:
            start_t = start_timestamp[-1]
            end_t = end_timestamp[-1]
        elif ack == False and failed_num == 0:
            start_t = start_timestamp[-1]
            end_t = end_timestamp[-1]
        else:
            start_t = start_timestamp[0]
            end_t = end_timestamp[0]
        #Compute the TOF and Slant Range
        self.dprint('StartTime: ' + str(start_timestamp))
        self.dprint('EndTime: ' + str(end_timestamp))
        self.dprint(ack)
        self.dprint(failed_num)
        tof_us = end_t - start_t
        #inc_time = 292270.667 #Time that remote modem need to send the ack (4 caracters).
        inc_time = 265646.667 #Time that remote modem need to send the ack (1 caracter).
        slant_range = (tof_us-inc_time)/2./1e6 * SOUND_SPEED
        self.dprint('SlantRange = %.2f m'%slant_range)
        '''
        #Obtain the TOF using the AT?T command, which will substitute the previouse method
        if ack == False:
        	self.dprint('Range error occurred')
        	return -1
        tof_us = -1
        for i in range(2):
        	#try 2 times to get the range if not, return -1
        	try:
        		tof_us = int(self.send(b'AT?T'))
        		break
        	except:
        		rospy.sleep(.1)
        if tof_us == -1:
        	print('ERROR: AT?T could not be sent')
        	return -1
        slant_range = tof_us/1e6 * SOUND_SPEED
        self.dprint('SlantRange = %.2f m'%slant_range)
        return slant_range


	
	
	
	
























        
