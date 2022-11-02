#!/usr/bin/env python
# license removed for brevity
import serial
import rospy
import numpy as np
from stalker.srv import EnableGoToWatch, MeasureRange, DisableGoToWatch
from cola2_msgs.msg import NavSts
import sensor_msgs.msg as sensor_msgs
import std_msgs.msg as std_msgs
from geometry_msgs.msg import Point, PointStamped, Vector3Stamped
from math import pow, atan2, sqrt
from utils import Target
import sys 
import utm
import time
from datetime import datetime

from stalker.srv import EnableGoToWatch
from cola2_msgs.srv import Goto

#for RL algorithm

#from rl_algorithms.maddpg import MADDPG
#from rl_algorithms.matd3_bc import MATD3_BC
#from rl_algorithms.masac import MASAC
#from pretrined_rl_network import rl_agent
#import torch

import os
from configparser import ConfigParser

import np_agent


TARGET_DISTANCE_THRESHOLD = 1. #to update the go_to_watch
USE_EVOLOGICS_EMULATOR = True


class TargetTracking:
    def __init__(self,target_estimation_method = 'ls', path_method = 'circumference', pretrined_agent = ''):
        # Creates a node with name 'stalker' and make sure it is a
        # unique node (using anonymous=True).
        rospy.init_node('stalker', anonymous=True)
        # A subscriber to the topic '/sparus2/navigator/navigation'. self.update_pose is called
        # when a message of type Pose is received.
        self.pose_subscriber = rospy.Subscriber('/sparus2/navigator/navigation',NavSts,self.update_pose)
        # A subscriber to the topic 'stalker/target_position'. self.update_real_target_pos is called
        # when a message of type Pose is received.
        self.target_subscriber = rospy.Subscriber('stalker/target_position',PointStamped,self.update_real_target_pos)
        
        if USE_EVOLOGICS_EMULATOR == True:
        	# Wait until a service is crated and save it to measure range from acoustic modem
        	rospy.wait_for_service('stalker/measure_range')
        	try:
        		self.measure_range = rospy.ServiceProxy('stalker/measure_range',MeasureRange)
        	except rospy.ServiceException as e:
        		print("Service call failed: %s"%e)
                
        # Wait until a service is crated and save it to send go to watch requests
        rospy.wait_for_service('stalker/enable_go_to_watch')
        try:
                self.go_to_watch = rospy.ServiceProxy('stalker/enable_go_to_watch',EnableGoToWatch)
        except rospy.ServiceException as e:
                print("Service call failed: %s"%e)
        
        # Wait until a service is crated and save it to send go to watch requests
        rospy.wait_for_service('stalker/disable_go_to_watch')
        try:
                self.disable_go_to_watch = rospy.ServiceProxy('stalker/disable_go_to_watch',DisableGoToWatch)
        except rospy.ServiceException as e:
                print("Service call failed: %s"%e)
                
        # This maybe should be done inside a go_to_angle_service.py
        # Wait til service is created					
        rospy.wait_for_service('/sparus2/captain/enable_goto')
        try:
        	self.enable_goto = rospy.ServiceProxy('/sparus2/captain/enable_goto',Goto)        
        except rospy.ServiceException as e:
        	print("Service call failed: %s"%e)
           
        	
                
        # A publisher to publish pointcloud based on Particle Filters
        self.point_cloud = rospy.Publisher('stalker/target_point_cloud', sensor_msgs.PointCloud2, queue_size=1)
        # A publisher to publish the estimated target position
        self.p_estimated_target_position = rospy.Publisher('stalker/estimated_target_position', PointStamped, queue_size=10)
        # A publisher to publish the last range measured
        self.last_range_measured = rospy.Publisher('stalker/range_measured', sensor_msgs.Range, queue_size=10)
        # A publisher to publish the last range measured
        self.set_ocean_current = rospy.Publisher('/sparus2/dynamics/current', Vector3Stamped, queue_size=1)


        self.pose = NavSts()
        self.r_targ_pos = PointStamped()
        #initial AUV position variables
        self.depth = 4.
        self.auv_position = [0., 0., 0., 0.]
        self.current_yaw = 0.
        #initial position of the target
        self.real_target_x = 0.
        self.real_target_y = 0.
        
        self.dnn = pretrined_agent
        self.auv_origin_x = 0.
        self.auv_origin_y = 0.
        
        self.origin_easting = 0.
        self.origin_northing = 0.
        
        #debug options
        self.debug_print = True
        
        #Target estimation method
        self.target_estimation_method = target_estimation_method       
        
        #Tracking method
        self.path_method = path_method
        self.target_yaw = 0.
        if self.path_method == 'rl':
        	#load the a new agent using a class:
            #self.trained_rl_agent2 = rl_agent(pretrined_agent)
            if pretrined_agent != 'qmix':
            	self.trained_rl_agent = np_agent.np_rl_agent() # A SAC agent trained with configuration: MASACa_l_v2
            else:
            	self.trained_rl_agent = np_agent.np_rnn_rl_agent() # A H-LSTM-SAC agent trained with configuration: MASACQMIX_lstm_emofish


    def update_pose(self, data):
        """Callback function which is called when a new message of type NavSts is
        received by the subscriber."""
        self.pose = data
        lat = self.pose.global_position.latitude
        lon = self.pose.global_position.longitude
        #AUV current position in UTM format
        tuple = utm.from_latlon(lat, lon)
        easting, northing, zonenumber, zoneletter = tuple
        #origin
        lat = self.pose.origin.latitude
        lon = self.pose.origin.longitude
        #AUV current position in UTM format
        tuple = utm.from_latlon(lat, lon)
        self.origin_easting, self.origin_northing, zonenumber, zoneletter = tuple
        #Update AUV position variables 
        self.auv_position[0] = easting - self.origin_easting
        self.auv_position[2] = northing - self.origin_northing
        self.auv_position[1] = self.pose.body_velocity.x
        self.auv_position[3] = self.pose.body_velocity.y
        self.depth = self.pose.position.depth
        self.current_yaw = self.pose.orientation.yaw
        return
        
    def update_real_target_pos(self, data):
        """Callback function which is called when a new real target position of type PointStamped is
        received by the subscriber."""
        self.r_targ_pos = data
        self.real_target_x = self.r_targ_pos.point.x
        self.real_target_y = self.r_targ_pos.point.y
        return
        
        
    def start(self,target, modem_id, radius, distance_between_meas, num_points, ocean_current=0, ocean_current_direction=0, final_depth=0, final_altitude=0, surge_velocity=0):
        """
        1- Estimates the target position using range-only method based on Particle Filter or Least Squares
        2- Move the AUV arount the target estimation using a circunference with determined radius
        """
        old_auv_position = list(self.auv_position)
        old_target_position = [0., 0., 0., 0.,] #[x,vx,y,vy]
        iteration = 0
        auv_distance_threshold = distance_between_meas + 0
        self.dprint('Distance between each range measurement is %.2f m'%auv_distance_threshold)
        last_time = rospy.get_time()
        #Start searching using the initial AUV position
        rospy.sleep(1.)
        self.go_to_watch(self.auv_position[0],self.auv_position[2],radius,6,True)
        
        #Initialize the estimated target position with current initial AUV position
        target.position[0] = self.auv_position[0]
        target.position[2] = self.auv_position[2]
        
        #Init point cloud and estimated target position publishers
        self.publish_estimated_position(target,target.position[0],target.position[2])
        
        #Set ocean currents
        print("Set ocean currents into the environment: Current=%.3f, Direction=%.3f"%(ocean_current,ocean_current_direction))
        current = Vector3Stamped()
        current.header.seq = 1
        current.header.frame_id = "world_ned"
        current.vector.x = np.cos(ocean_current_direction*np.pi/180.)*ocean_current
        current.vector.y = np.sin(ocean_current_direction*np.pi/180.)*ocean_current
        current.vector.z = 0
        self.set_ocean_current.publish(current)
        print("Done")
        
        #datatime for log purposes
        now = datetime.now() # current date and time
        date_time_folder = now.strftime("%m%d%YT%H%M%S")
        
        timestamp_t = []
        auv_x_t = []
        auv_y_t = []
        auv_vx_t = []
        auv_vy_t = []
        real_target_x_t = []
        real_target_y_t = []
        p_target_x_t = []
        p_target_y_t = []
        range_t = []
        new_range_flag_t = []
        origin_northing_t = []
        origin_easting_t = []
        angle_waypoint_x_t = []
        angle_waypoint_y_t = []
        new_range_flag = -1
        angle_waypoint_x = 0.
        angle_waypoint_y = 0.
        while  not rospy.is_shutdown():
                try:
                    #Distance traveled since last time
                    auv_distance = np.sqrt((self.auv_position[0]-old_auv_position[0])**2+(self.auv_position[2]-old_auv_position[2])**2)
                    if auv_distance > auv_distance_threshold or (rospy.get_time()-last_time) > 60.:
                        #Measure a new range using range_meas_service
                        if USE_EVOLOGICS_EMULATOR == True:
                            for mod_id in modem_id[::-1]:
                                print('READING RANGE FROM MODEM ID: ',mod_id)
                                    try:
                                    	info = self.measure_range(mod_id)
                                        slant_range = info.slant_range
                                    except:
                                        slant_range = -1
                                        print('WARNING: PORT READ ERROR, RECURSIVE ACCESS')
                                self.dprint('New range measured: %.2f m'%slant_range)
                                time.sleep(5)
                        else:
                            slant_range = np.sqrt((self.real_target_x-self.auv_position[0])**2+(self.real_target_y-self.auv_position[2])**2)
                            slant_range += np.random.normal(0.,1.)
                            self.dprint('New range measured: %.2f m'%slant_range)
                        
                        if slant_range == -1:
                            new_range = False
                            new_range_flag = -1
                        else:
                            new_range = True
                            new_range_flag = 1
                            self.auv_origin_x = self.auv_position[0]
                            self.auv_origin_y = self.auv_position[2]
                 
                        #Publish the last range measured
                        header = std_msgs.Header(frame_id='world_ned',stamp=rospy.Time.now())
                        last_range = sensor_msgs.Range(header=header, field_of_view = 0.1,  range= slant_range)
                        self.last_range_measured.publish(last_range)
                        
                        if self.target_estimation_method == 'pf':
                            #Update the estimated target position using a Particle Filter 
                            target.updatePF(dt=rospy.get_time()-last_time, new_range=new_range, z=slant_range, myobserver=self.auv_position, update=new_range)
                        else:
                            #Update the estimated target position using a Least Square method
                            target.updateLS(dt=rospy.get_time()-last_time, new_range=new_range, z=slant_range, myobserver=self.auv_position)
                        
                        last_time = rospy.get_time()
                        self.dprint('New estimated target position at x=%.2f m, y=%.2f m'%(target.position[0],target.position[2]))
                        # Publish the estimated target position
                        #self.publish_estimated_position(target,target.pf.x.T[0],target.pf.x.T[2])
                        self.publish_estimated_position(target,target.position[0],target.position[2])
                        
                
                        #See if target estimation has changed significantly
                        target_distance = np.sqrt((target.position[0]-old_target_position[0])**2+(target.position[2]-old_target_position[2])**2)
                        if target_distance > TARGET_DISTANCE_THRESHOLD or self.path_method == 'rl':
                            #Move the AUV using standard Circle path:
                            if self.path_method == 'rl':
                                self.disable_go_to_watch()
                                #Send the AUV using the angle (yaw) action generated by the deep RL network
                                if slant_range == -1:
                                    #compute the range from the last target estimated position
                                    slant_range = np.sqrt((target.position[0]-self.auv_position[0])**2+(target.position[2]-self.auv_position[2])**2)
                                #get action
                                obs = [float(self.auv_position[1])/1000.,
                                        float(self.auv_position[3])/1000.,
                                        float(self.auv_position[0])/1000.,
                                        float(self.auv_position[2])/1000.,
                                        float(target.position[0]-self.auv_position[0])/1000., 
                                        float(target.position[2]-self.auv_position[2])/1000., 
                                        float(slant_range)/1000.]
                                if self.dnn == 'qmix':
                                    obs_fake_speed = [float(np.cos(self.current_yaw))*0.3,
                                                    float(np.sin(self.current_yaw))*0.3,
                                                    float(self.auv_position[0])/50.,
                                                    float(self.auv_position[2])/50.,
                                                    float(target.position[0]-self.auv_position[0])/50., 
                                                    float(target.position[2]-self.auv_position[2])/50., 
                                                    float(slant_range)/50.,
                                                    float(15.)/50.,
                                                    float(self.auv_origin_x)/50.,
                                                    float(self.auv_origin_y)/50.] 
                                else:
                                    obs_fake_speed = [float(np.cos(self.current_yaw))*0.3,
                                                    float(np.sin(self.current_yaw))*0.3,
                                                    float(self.auv_position[0])/50.,
                                                    float(self.auv_position[2])/50.,
                                                    float(target.position[0]-self.auv_position[0])/50., 
                                                    float(target.position[2]-self.auv_position[2])/50., 
                                                    float(slant_range)/50.] 
                                #0.03 is the speed used to train the model
                                #action = self.trained_rl_agent.next_action(obs)
                                action = self.trained_rl_agent.next_action(obs_fake_speed)
                                #action2 = self.trained_rl_agent2.next_action(obs_fake_speed)
                                #print('NEW deepRL action1=%.2f, action2=%.2f'%(action,action2))
                                print('NEW deepRL action=',action)
                                print('obs              =', obs_fake_speed)
                                inc_angle = action * 0.3 #we multiply by 0.3 to limit the minimum angle that the AUV can do
                                self.target_yaw  = self.target_yaw + inc_angle
                                if self.target_yaw  > np.pi*2.:
                                    self.target_yaw -= np.pi*2.
                                if self.target_yaw  < -np.pi*2:
                                    self.target_yaw  += np.pi*2.
                                angle_waypoint_x = np.cos(self.target_yaw ) * 30.
                                angle_waypoint_y = np.sin(self.target_yaw ) * 30.
                                info2 = self.enable_goto(final_x=self.auv_position[0]+angle_waypoint_x,
                                                         final_y=self.auv_position[2]+angle_waypoint_y,
                                                          final_depth = final_depth,
                                                           final_altitude = final_altitude,
                                                            reference = 0,
                                                             heave_mode = 0,
                                                              surge_velocity = surge_velocity,
                                                               tolerance_xy = 3,
                                                                timeout = 1800,
                                                                 no_altitude_goes_up = True)
                                                                    old_target_position = target.position
                            else: # path_method == 'Circle'
                                #Move the center of the circunference using go_to_watch_service
                                if abs(old_target_position[0]-target.position[0])<500 and abs(old_target_position[2]-target.position[2])<500:
                                    self.go_to_watch(target.position[0],target.position[2],radius,6,True)
                                else:
                                    self.go_to_watch(200,200.,radius,6,True)
                                old_target_position = target.position
                        
                        
                            #save data for post-processing
                            header = 'timestamp auv_x auv_y auv_vx auv_vy real_target_x real_target_y p_target_x p_target_y range true_range origin_northing origin_easting wp_x wp_y'
                            timestamp_t.append(time.time())
                            auv_x_t.append(self.auv_position[0])
                            auv_y_t.append(self.auv_position[2])
                            auv_vx_t.append(self.auv_position[1])
                            auv_vy_t.append(self.auv_position[3])
                            real_target_x_t.append(self.real_target_x)
                            real_target_y_t.append(self.real_target_y)
                            p_target_x_t.append(target.position[0])
                            p_target_y_t.append(target.position[2])
                            range_t.append(slant_range)
                            new_range_flag_t.append(new_range_flag)
                            origin_northing_t.append(self.origin_northing)
                            origin_easting_t.append(self.origin_easting)
                            angle_waypoint_x_t.append(self.auv_position[0]+angle_waypoint_x)
                            angle_waypoint_y_t.append(self.auv_position[2]+angle_waypoint_y)
                            aux_t = np.concatenate((np.matrix(timestamp_t),
                                        np.matrix(auv_x_t),
                                        np.matrix(auv_y_t),
                                        np.matrix(auv_vx_t),
                                        np.matrix(auv_vy_t),
                                        np.matrix(real_target_x_t),
                                        np.matrix(real_target_y_t),
                                        np.matrix(p_target_x_t),
                                        np.matrix(p_target_y_t),
                                        np.matrix(range_t),
                                        np.matrix(new_range_flag_t),
                                        np.matrix(origin_northing_t),
                                        np.matrix(origin_easting_t),
                                        np.matrix(angle_waypoint_x_t),
                                        np.matrix(angle_waypoint_y_t)),axis=0)
                            path_to_save = os.path.dirname(os.getcwd())+'/catkin_ws/src/stalker/src/'
                            np.savetxt(path_to_save+'log_test_'+date_time_folder+'.txt', aux_t.T, fmt="%.6f", header=header, delimiter =',') 
                            
                            #See if the AUV is on the circle (i.e. is conducting the circunference, not going to it)
                            auv_distance_from_target = np.sqrt((self.auv_position[0]-target.position[0])**2+(self.auv_position[2]-target.position[2])**2) 
                            if auv_distance_from_target>(radius-5) and auv_distance_from_target<(radius+5):
                                    if num_points == -1:
                                            self.dprint('Iteration '+str(iteration)+'/--')
                                    else:
                                            self.dprint('Iteration '+str(iteration)+'/'+str(int(num_points+2)))
                                    if iteration > num_points+1 and num_points != -1:
                                            self.disable_go_to_watch()
                                            break
                                    iteration += 1
                            old_auv_position = list(self.auv_position)
                
                    #Set a sleep time to save computer resources
                    rospy.sleep(.1)
                except rospy.ROSInterruptException or KeyboardInterrupt:
                        print('KeyboardInterrupt')
                        self.disable_go_to_watch()
                        return
                        
        return
    
    def publish_estimated_position(self,target,x,y):
        '''
        Publish the estimated target position using Particle Filter or Least Squares
        '''
        if self.target_estimation_method == 'pf':
        	# make the 2D (xyzrgba) array
        	x = x.reshape([x.size,1])[::10]
        	y = y.reshape([y.size,1])[::10]
        	z = np.zeros(x.shape)
        	r = np.ones(x.shape)
        	g = np.zeros(x.shape)
        	b = np.zeros(x.shape)
        	a = np.zeros(x.shape)+0.3
        	points = np.concatenate((x,y,z,r,g,b,a),axis=1)
        	ros_dtype = sensor_msgs.PointField.FLOAT32
        	dtype = np.float32
        	itemsize = np.dtype(dtype).itemsize
        	
        	data = points.astype(dtype).tobytes()
        	
        	fields = [sensor_msgs.PointField(name=n, offset=i*itemsize, datatype=ros_dtype, count=1) for i, n in enumerate('xyzrgba')]
        	
        	header = std_msgs.Header(frame_id='world_ned',stamp=rospy.Time.now())
        	
        	pc2 = sensor_msgs.PointCloud2(
		                        header=header,
		                        height=1,
		                        width=points.shape[0],
		                        is_dense=False,
		                        is_bigendian=False,
		                        fields=fields,
		                        point_step=(itemsize*7),
		                        row_step=(itemsize*7*points.shape[0]),
		                        data=data)
        	#Publish the particles positions as a point cloud
        	self.point_cloud.publish(pc2)
        
        #Publish the estimated target posiiton
        # Create the point message
        point = PointStamped()
        point.header.stamp = rospy.Time.now()
        point.header.frame_id = "world_ned"
        point.point.x = target.position[0]
        point.point.y = target.position[2]
        self.p_estimated_target_position.publish(point)
        
        return
        
    def dprint(self,message):
        if self.debug_print == True:
            print(message)
        return
    
        
                
        
        
   
if __name__ == '__main__':
    try:
        try:
            modem_id = (sys.argv[1])
            radius = float(sys.argv[2])
            distance_between_meas = int(sys.argv[3])
            num_points = float(sys.argv[4])
            target_estimation_method = sys.argv[5]
            path_method = sys.argv[6]
            pretrined_agent = sys.argv[7]
            ocean_current = float(sys.argv[8])
            ocean_current_direction = float(sys.argv[9])
            final_depth = float(sys.argv[10])
            final_altitude = float(sys.argv[11])
            surge_velocity = float(sys.argv[12])
            
            print('method=',target_estimation_method)
            
        except IndexError:
            print('Three arguments are requestd <modem ID>, <circumference radius>, <distance between measurements> and <number of range measurements>')
            sys.exit()
        if radius<1 or radius>500:
            print('Radius must be between 1 and 500')
            sys.exit()
        elif num_points<1 or num_points>100:
            if num_points == -1:
                print('Infinit number of points will be measured')
            else:
                print('Number of measurement must be between 1 and 100')
                sys.exit()
        elif distance_between_meas<1 or distance_between_meas>100:
            print('Distance between measurements must be between 1 and 100')
            sys.exit()
        target = Target()
        tracker = TargetTracking(target_estimation_method, path_method, pretrined_agent)
        tracker.start(target, np.array(modem_id.split(','),dtype=int), radius, distance_between_meas, num_points, ocean_current, ocean_current_direction, final_depth, final_altitude, surge_velocity)
        print('Done')

    except rospy.ROSInterruptException or KeyboardInterrupt:
        pass
