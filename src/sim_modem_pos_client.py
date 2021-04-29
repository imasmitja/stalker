#!/usr/bin/env python
# license removed for brevity
import rospy
import sys 
from stalker.srv import SimModemPosition

def move_target_modem(x,y,z):
	rospy.wait_for_service('stalker/sim_modem_pos')
	try:
		move_to_point = rospy.ServiceProxy('stalker/sim_modem_pos',SimModemPosition)
		resp = move_to_point(x,y,z)
		return resp
	except rospy.ServiceException as e:
		print("Service call failed: %s"%e)

   
if __name__ == '__main__':
	try:
		try:
			x = float(sys.argv[1])
			y = float(sys.argv[2])
		except IndexError:
			print('Two arguments are requestd <modem x position> and <modem y position>')
			sys.exit()
		if x<0 or x>1000:
			print('X must be between 0 and 1000')
			sys.exit()
		elif y<0 or y>1000:
			print('Y must be between 0 and 1000')
			sys.exit()
		print(move_target_modem(x,y,0))
		sys.exit()
	except rospy.ROSInterruptException:
		pass
