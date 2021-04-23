#!/usr/bin/env python
# license removed for brevity
import rospy
import numpy as np
from stalker.srv import EnableGoToWatch, MeasureRange, DisableGoToWatch
from cola2_msgs.msg import NavSts
import sensor_msgs.msg as sensor_msgs
import std_msgs.msg as std_msgs
from geometry_msgs.msg import Point, PointStamped
from math import pow, atan2, sqrt
from utils import Target
import sys 
import utm
import time


TARGET_DISTANCE_THRESHOLD = 1. #to update the go_to_watch


class TargetTracking:
    def __init__(self):
    	# Creates a node with name 'stalker' and make sure it is a
    	# unique node (using anonymous=True).
    	rospy.init_node('stalker', anonymous=True)
    	# A subscriber to the topic '/sparus2/navigator/navigation'. self.update_pose is called
	# when a message of type Pose is received.
    	self.pose_subscriber = rospy.Subscriber('/sparus2/navigator/navigation',NavSts,self.update_pose)
    	
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
    		
    	# A publisher to publish pointcloud based on Particle Filters
    	self.point_cloud = rospy.Publisher('stalker/target_point_cloud', sensor_msgs.PointCloud2, queue_size=1)
    	# A publisher to publish the estimated target position
    	self.p_estimated_target_position = rospy.Publisher('stalker/estimated_target_position', PointStamped, queue_size=10)
    	
    	self.pose = NavSts()
    	#initial AUV position variables
    	self.depth = 0.
    	self.auv_position = [0., 0., 0., 0.]
    	
    	#debug options
    	self.debug_print = True


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
        origin_easting, origin_northing, zonenumber, zoneletter = tuple
        #Update AUV position variables
        self.auv_position[0] = northing - origin_northing
        self.auv_position[2] = easting - origin_easting
        self.auv_position[1] = self.pose.body_velocity.x
        self.auv_position[3] = self.pose.body_velocity.y
        self.depth = self.pose.position.depth
        return
    	
    def start(self,target, modem_id, radius, distance_between_meas, num_points):
    	"""
    	1- Estimates the target position using range-only method based on Particle Filter
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
    	#Init point cloud and estimated target position publishers
    	self.publish_estimated_position(target,target.pf.x.T[0],target.pf.x.T[2])
    	while  not rospy.is_shutdown():
    		try:
    			#Distance traveled since last time
        		auv_distance = np.sqrt((self.auv_position[0]-old_auv_position[0])**2+(self.auv_position[2]-old_auv_position[2])**2)
        		if auv_distance > auv_distance_threshold:
        	
        			#Measure a new range using range_meas_service
        			info = self.measure_range(modem_id)
        			self.dprint('New range measured: %.2f m'%info.slant_range)
        		
        			#Update the estimated target position using a Particle Filter
        			target.updatePF(dt=rospy.get_time()-last_time, new_range=True, z=info.slant_range, myobserver=self.auv_position, update=True)
        			last_time = rospy.get_time()
        			self.dprint('New estimated target position at x=%.2f m, y=%.2f m'%(target.position[0],target.position[2]))
        			# Publish the estimated target position
        			self.publish_estimated_position(target,target.pf.x.T[0],target.pf.x.T[2]) 
        			
        		
        			#See if target estimation has changed significantly
        			target_distance = np.sqrt((target.position[0]-old_target_position[0])**2+(target.position[2]-old_target_position[2])**2)
        			if target_distance > TARGET_DISTANCE_THRESHOLD:
        		
        				#Move the center of the circunference using go_to_watch_service
        				self.go_to_watch(target.position[0],target.position[2],radius,6,True)
        				old_target_position = target.position
        			
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
    			return
    			
    	return
    
    def publish_estimated_position(self,target,x,y):
    	'''
    	Publish the estimated target position using Particle Filter
    	'''
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
    	point.point.x = target.pf._x[0]
    	point.point.y = target.pf._x[2]
    	self.p_estimated_target_position.publish(point)
    	
    	return
    	
    def dprint(self,message):
        if self.debug_print == True:
            print(message)
        return
    
    	
        	
        
        
   
if __name__ == '__main__':
    try:
        try:
            modem_id = int(sys.argv[1])
            radius = float(sys.argv[2])
            distance_between_meas = int(sys.argv[3])
            num_points = float(sys.argv[4])
            
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
        tracker = TargetTracking()
        tracker.start(target, modem_id, radius, distance_between_meas, num_points)
        print('Done')

    except rospy.ROSInterruptException or KeyboardInterrupt:
        pass
