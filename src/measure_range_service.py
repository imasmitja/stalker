#!/usr/bin/env python
# license removed for brevity
import rospy
import numpy as np
import sys 
from stalker.srv import MeasureRange
from utils import netcat
from cola2_msgs.msg import NavSts
import utm


HOSTNAME1 = '10.42.57.1'
HOSTNAME2 = '10.42.57.2'
PORT1 = 9200
PORT2 = 11000
SERIAL_PORT_NAME = '/dev/ttyS10'

class SlantRange:

    def __init__(self,debug,sim,interface):
        # Creates a node with name 'measure_range' and make sure it is a
        # unique node (using anonymous=True).
        rospy.init_node('slant_range', anonymous=True)
        
        # A service
        self.srv = rospy.Service('stalker/measure_range', MeasureRange,
        					self.measure_range_callback)
        
        # A subscriber to the topic '/sparus2/navigator/navigation'. self.update_pose is called
        # when a message of type Pose is received.
        self.pose_subscriber = rospy.Subscriber('/sparus2/navigator/navigation',
                                                NavSts, self.update_pose)
        
        #I nitialize the acoustic modem
        print('Initializing SparusII modem...')
        self.modem = netcat(HOSTNAME1,PORT1,'sparus2_modem',debug=debug,interface = interface, serial_port_name = SERIAL_PORT_NAME, sim = sim)
        print('Modem initialized')
        self.sim_modem_mov = sim
        self.auv_x = 0
        self.auv_y = 0
        self.auv_z = 0
        self.pose = NavSts()
        
        
    def measure_range_callback(self, request):
    	#send a message to the remote acoustic modem to compute the time of flight and obtain the slant range
    	#slant_range =  100. #this is fake, used only for debuging purposes
    	if self.sim_modem_mov == True:
    		self.modem.move(self.auv_x,self.auv_y,self.auv_z)
    	slant_range = self.modem.slant_range(request.modem_id)
    	return slant_range
    
    def update_pose(self, data):
        """Callback function which is called when a new message of type NavSts is
        received by the subscriber."""
        self.pose = data
        lat = self.pose.global_position.latitude
        lon = self.pose.global_position.longitude
        self.auv_z = self.pose.position.depth
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
        self.auv_x = easting-origin_easting
        self.auv_y = northing-origin_northing
        return

   
if __name__ == '__main__':
    try:
    	try:
    		debug = (sys.argv[1])
    		sim = (sys.argv[2])
    		interface = (sys.argv[3])
    		if debug == 'True':
    			debug = True
    		elif debug == 'False':
    			debug = False
    		else:
    			print('<debug> must be True or False')
    			sys.exit()
    		if sim == 'True':
    			sim = True
    		elif sim == 'False':
    			sim = False
    		else:
    			print('<sim> must be True or False')
    			sys.exit()
    		if interface != 'serial' and interface != 'ethernet':
    			print('<interface> must be serial or ethernet')
    			sys.exit()
    	except IndexError:
    		debug = False
    		sim = False
    		interface = 'serial'
    		print('Arguments can be passed for debuging and simulating purposes, if not, default will be used')
    	
    	done = SlantRange(debug=debug,sim=sim,interface=interface)
    	rospy.spin()
    except rospy.ROSInterruptException:
    	pass
