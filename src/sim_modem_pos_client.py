#!/usr/bin/env python
# license removed for brevity
import rospy
import sys 
from stalker.srv import SimModemPosition
import numpy as np

YAWN_MOVEMENT = False

def move_target_modem(x,y,z,v,yaw):
	rospy.wait_for_service('stalker/sim_modem_pos')
	iteration = 0
	try:
		while  not rospy.is_shutdown():
			if YAWN_MOVEMENT == False: #linear movement on if velocity is != than 0
				move_to_point = rospy.ServiceProxy('stalker/sim_modem_pos',SimModemPosition)
				resp = move_to_point(x,y,z)
				rospy.sleep(1.)
				x += v*np.cos(yaw)
				y += v*np.sin(yaw)
			else: #yawn movement on
				move_to_point = rospy.ServiceProxy('stalker/sim_modem_pos',SimModemPosition)
				resp = move_to_point(x,y,z)
				rospy.sleep(1.)
				x += v*np.cos(yaw)
				y += v*np.sin(yaw)
				iteration += 1
				const = 6000
				if iteration == const*1 + (const/4)*0:
					yaw += np.pi/2.
				if iteration == const*1 + (const/4)*1:
					yaw += np.pi/2.
				if iteration == const*2 + (const/4)*1:
					yaw -= np.pi/2.
				if iteration == const*2 + (const/4)*2:
					yaw -= np.pi/2.
				if iteration == const*3 + (const/4)*2:
					yaw += np.pi/2.
				if iteration == const*3 + (const/4)*3:
					yaw += np.pi/2.
				if iteration == const*4 + (const/4)*3:
					yaw -= np.pi/2.
				if iteration == const*4 + (const/4)*4:
					yaw -= np.pi/2.
				if iteration == const*5 + (const/4)*4:
					yaw += np.pi/2.
				if iteration == const*5 + (const/4)*5:
					yaw += np.pi/2.
				if iteration == const*6 + (const/4)*5:
					yaw -= np.pi/2.
				if iteration == const*6 + (const/4)*6:
					yaw -= np.pi/2.
				
				
			
			
		return resp
	except rospy.ServiceException as e:
		print("Service call failed: %s"%e)

   
if __name__ == '__main__':
	try:
		try:
			x = float(sys.argv[1])
			y = float(sys.argv[2])
			z = float(sys.argv[3])
			v = float(sys.argv[4])
			yaw = float(sys.argv[5])
		except IndexError:
			print('Two arguments are requestd <modem x position> and <modem y position>')
			sys.exit()
		if x<-1000 or x>1000:
			print('X must be between -1000 and 1000')
			sys.exit()
		elif y<-1000 or y>1000:
			print('Y must be between -1000 and 1000')
			sys.exit()
		elif z<0 or z>1000:
			print('Z must be between 0 and 1000')
			sys.exit()
		elif v<0 or v>2:
			print('V must be between 0 and 2')
			sys.exit()
		elif yaw<0 or yaw>360:
			print('YAW must be between 0 and 360')
			sys.exit()
		print(move_target_modem(x,y,z,v,yaw*np.pi/180.))
		sys.exit()
	except rospy.ROSInterruptException:
		pass
