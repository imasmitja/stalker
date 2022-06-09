#!/usr/bin/env python
# license removed for brevity
import rospy
import numpy as np
import sys 
from stalker.srv import SimModemPosition
from geometry_msgs.msg import PointStamped
from utils import netcat
import utm

HOSTNAME1 = '10.42.57.1'
HOSTNAME2 = '10.42.57.2'
PORT1 = 9200
PORT2 = 11000

class SimModemPos:
    def __init__(self,debug):
        # Creates a node with name 'measure_range' and make sure it is a
        # unique node (using anonymous=True).
        rospy.init_node('sim_modem_pos', anonymous=True)
        
        # A service
        self.srv1 = rospy.Service('stalker/sim_modem_pos', SimModemPosition, 
        							self.sim_modem_pos_callback)
        							
        # A publisher
        self.real_target_position = rospy.Publisher('stalker/target_position', PointStamped, queue_size=10)
        
        
        #I nitialize the acoustic modem
        #self.modem = netcat(HOSTNAME2,PORT1,'target_modem',debug=debug)
        return
        
    def sim_modem_pos_callback(self, request):
    	#Move the target modem to the desired position
    	#print('WARNING: Moving target modem to %.2f m(x) %.2f m(y) %.2f m(z)'%(request.x, request.y, request.z))
    	#self.modem.move(request.y,request.x,request.z)
    	# Create the point message
    	point = PointStamped()
    	point.header.stamp = rospy.Time.now()
    	point.header.frame_id = "world_ned"
    	point.point.x = request.x
    	point.point.y = request.y
    	point.point.z = request.z
    	self.real_target_position.publish(point)
    	
    	return True



if __name__ == '__main__':
    try:
    	try:
    		debug = (sys.argv[1])
    		if debug == 'True':
    			debug = True
    		elif debug == 'False':
    			debug = False
    		else:
    			print('<debug> must be True or False')
    			sys.exit()
    	except IndexError:
    		debug = False
    		print('Arguments can be passed for debuging purposes, if not, default will be used')
    	
    	done = SimModemPos(debug)
    	rospy.spin()
    except rospy.ROSInterruptException:
    	pass
