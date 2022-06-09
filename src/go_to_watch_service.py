#!/usr/bin/env python
# license removed for brevity
import rospy
import sys 
import numpy as np
from stalker.srv import EnableGoToWatch, DisableGoToWatch
from cola2_msgs.srv import Goto
from cola2_msgs.msg import CaptainStatus
from cola2_msgs.msg import NavSts
from std_srvs.srv import Trigger
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import utm
import threading
   	
class CircumferencePath:

    def __init__(self,final_depth, final_altitude, reference, heave_mode, surge_velocity, tolerance_xy, timeout, no_altitude_goes_up):
        # Creates a node with name 'turtlebot_controller' and make sure it is a
        # unique node (using anonymous=True).
        rospy.init_node('circumference_path', anonymous=True)
        
        # Publisher which will publish to the topic '/sparus2/captain/trajectory_path'.
        self.path_publisher = rospy.Publisher('/sparus2/captain/trajectory_path',
                                                  Path, queue_size=10)

        # A subscriber to the topic '/sparus2/captain/captain_status'. self.update_captain_status is called
        # when a message of type Pose is received.
        self.captain_status_subscriber = rospy.Subscriber('/sparus2/captain/captain_status',
                                                CaptainStatus, self.update_captain_status)
                                                
        # A subscriber to the topic '/sparus2/navigator/navigation'. self.update_pose is called
        # when a message of type Pose is received.
        self.pose_subscriber = rospy.Subscriber('/sparus2/navigator/navigation',
                                                NavSts, self.update_pose)
             
        # A service
        self.srv1 = rospy.Service('stalker/enable_go_to_watch', EnableGoToWatch,
        					self.enable_go_to_watch_callback)
        # A service
        self.srv2 = rospy.Service('stalker/disable_go_to_watch', DisableGoToWatch,
        					self.disable_go_to_watch_callback)
        
        # Wait til service is created					
        rospy.wait_for_service('/sparus2/captain/enable_goto')
        try:
        	self.enable_goto = rospy.ServiceProxy('/sparus2/captain/enable_goto',Goto)        
        except rospy.ServiceException as e:
        	print("Service call failed: %s"%e)
        	
        # Wait til service is created					
        rospy.wait_for_service('/sparus2/captain/disable_goto')
        try:
        	self.disable_goto = rospy.ServiceProxy('/sparus2/captain/disable_goto',Trigger)        
        except rospy.ServiceException as e:
        	print("Service call failed: %s"%e)
        
        self.waypoin_reached = True
        self.rate = rospy.Rate(10)
        self.auv_x = 0
        self.auv_y = 0
        self.stop_threads = False
        self.thread = []
        self.final_depth = final_depth
        self.final_altitude = final_altitude
        self.reference = reference
        self.heave_mode = heave_mode
        self.surge_velocity = surge_velocity
        self.tolerance_xy = tolerance_xy
        self.timeout = timeout
        self.no_altitude_goes_up = no_altitude_goes_up

        
    def enable_go_to_watch_callback(self, request):
    	# if a previous thread exist, we kill it
    	self.stop_threads = True
    	try:
    		self.thread.join()
    	except:
    		pass
    	# if a previous goto is enable, we disable it
    	info = self.disable_goto()
    	# we launch a new thread
    	self.thread = threading.Thread(target=self.move2goal, args=(request.x, request.y, request.radius, request.points, request.loop))
    	self.thread.start()
    	#self.move2goal(request.x, request.y, request.radius, request.points, request.loop)
    	return True
    
    def disable_go_to_watch_callback(self,request):
    	# if a previous thread exist, we kill it
    	self.stop_threads = True
    	try:
    		self.thread.join()
    	except:
    		pass
    	# if a previous goto is enable, we disable it
    	info = self.disable_goto()
    	return True
    	
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
        self.auv_y = easting-origin_easting
        self.auv_x = northing-origin_northing
        self.auv_yaw = self.pose.orientation.yaw
        return

    def update_captain_status(self, data):
        """Callback function which is called when a new message of type Pose is
        received by the subscriber."""
        if data.state == 0:
        	self.waipoint_reached = True
        else:
        	self.waipoint_reached = False
        return
        

    def move2goal(self,x_target,y_target,radius, points, continuous_loop):
        """
        Moves the AUV to the goal position and starts a circumference with the requested radius
        """
        self.stop_threads = False
        path = Path()
        num_waypoints = int(points)
        #First we need to create a set of waypoints to emulate a circumference
        angle_step = 2*np.pi/num_waypoints
        waypoints_x = []
        waypoints_y = []
        for i in range(num_waypoints+1):
        	waypoints_x.append(np.cos(-angle_step*i)*radius)
        	waypoints_y.append(np.sin(-angle_step*i)*radius)
        #Rotation matrix to put the 1st waypoint in front of the AUV
        #compute the distance between the AUV and the target position
        distance = np.sqrt((self.auv_x-x_target)**2+(self.auv_y-y_target)**2)
        #If the AUV is inside the path circumference, use its own yaw as angle of rotation, if not, use the angle comuted between themselves and the target
        if distance < radius+5: #was -5
        	#angle = self.auv_yaw
        	angle = np.arctan2((self.auv_y-y_target),(self.auv_x-x_target)) - angle_step
        else:
        	angle = np.arctan2((self.auv_y-y_target),(self.auv_x-x_target)) - angle_step
        waypoints_x_r = []
        waypoints_y_r = []
        for i in range(num_waypoints+1):
        	waypoints_x_r.append(waypoints_x[i]*np.cos(angle)-waypoints_y[i]*np.sin(angle))
        	waypoints_y_r.append(waypoints_x[i]*np.sin(angle)+waypoints_y[i]*np.cos(angle))
        #Translation to the target position
        waypoints_x_rt = x_target + np.array(waypoints_x_r)
        waypoints_y_rt = y_target + np.array(waypoints_y_r)
        #Publish the path
        path.header.stamp = rospy.Time()
        path.header.seq = num_waypoints
        path.header.frame_id = "world_ned"
        for i in range(num_waypoints+1):
        	pose = PoseStamped()
        	pose.header.seq = i
        	pose.header.frame_id = "world_ned"
        	pose.pose.position.x = waypoints_x_rt[i]
        	pose.pose.position.y = waypoints_y_rt[i]
        	path.poses.append(pose)
        self.path_publisher.publish(path)
        #Then we need to follow all the waypoints generated
        iteration_counter = 0
        while (True):
        	if self.waipoint_reached == True:
        		info = self.enable_goto(final_x=waypoints_x_rt[iteration_counter],
        					 final_y=waypoints_y_rt[iteration_counter],
        					  final_depth = self.final_depth,
        					   final_altitude = self.final_altitude,
        					    reference = self.reference,
        					     heave_mode = self.heave_mode,
        					      surge_velocity = self.surge_velocity,
        					       tolerance_xy = self.tolerance_xy,
        					        timeout = self.timeout,
        					         no_altitude_goes_up = self.no_altitude_goes_up)
        		iteration_counter += 1
        	if iteration_counter >= num_waypoints+1:
        		if continuous_loop == True:
        			iteration_counter = 0
        		else:
        			break
        	if self.stop_threads == True:
        		break
        	# Check at the desired rate.
        	self.rate.sleep()
        path = Path()
        path.header.seq = 0
        path.header.frame_id = "world_ned"
        pose = PoseStamped()
        pose.header.seq = 0
        pose.header.frame_id = "world_ned"
        pose.pose.position.x = 0
        pose.pose.position.y = 0
        path.poses.append(pose)
        self.path_publisher.publish(path)
        return  
   
if __name__ == '__main__':
    try:
    	try:
    		final_depth = float(sys.argv[1])
    		final_altitude = float(sys.argv[2])
    		reference = int(sys.argv[3])
    		heave_mode = int(sys.argv[4])
    		surge_velocity = float(sys.argv[5])
    		tolerance_xy = float(sys.argv[6])
    		timeout = float(sys.argv[7])
    		no_altitude_goes_up = (sys.argv[8])
    	except IndexError:
    		print('Three arguments are requestd <final_depth>, <final_altitude>, <reference>, <heave_mode>, <surge_velocity>, <tolerance_xy>, <timeout>, <no_altitude_goes_up>')
    		sys.exit()
    	if no_altitude_goes_up == 'True':
    		no_altitude_goes_up = True
    	elif no_altitude_goes_up == 'False':
    		no_altitude_goes_up = False
    	else:
    		print('<no_altitude_goes_up> must be True or False')
    		sys.exit()
    	done = CircumferencePath(final_depth, final_altitude, reference, heave_mode, surge_velocity, tolerance_xy, timeout, no_altitude_goes_up)
    	rospy.spin()

    except rospy.ROSInterruptException:
        pass
