#!/usr/bin/env python
# license removed for brevity
import rospy
import numpy as np
import sys
from stalker.srv import MeasureRange
from cola2_msgs.msg import NavSts

# modem position relative to first nav (in world frame)
RELATIVE_X = 0.0
RELATIVE_Y = 0.0
ABSOLUTE_Z = 5.0


class SlantRange:
    def __init__(self):
        # init node
        rospy.init_node('slant_range')

        # init vars
        self.auv_x = 0.0
        self.auv_y = 0.0
        self.auv_z = 0.0
        self.modem_x = 0.0
        self.modem_y = 0.0
        self.modem_z = 0.0
        self.has_nav = False

        # range measure service
        self.srv_range = rospy.Service('stalker/measure_range', MeasureRange,
        			            	   self.srv_measure_range)

        # subscriber to navigation
        self.sub_nav = rospy.Subscriber('/sparus2/navigator/navigation',
                                        NavSts, self.cbk_pose)

    def srv_measure_range(self, req):
        # check
        if not self.has_nav:
            rospy.logwarn("navigation not yet available, no range measured")
            return -1
        # otherwise
        rng = np.linalg.norm(
            np.array((self.auv_x, self.auv_y, self.auv_z)) -
            np.array((self.modem_x, self.modem_y, self.modem_z))
        )
        rospy.logwarn("range measurement {:.2f}".format(rng))
    	return rng

    def cbk_pose(self, msg):
        """Save current auv position and init modem at first message."""
        self.auv_x = msg.position.north
        self.auv_y = msg.position.east
        self.auv_z = msg.position.depth
        # init modem position
        if not self.has_nav:
            self.modem_x = self.auv_x + RELATIVE_X
            self.modem_y = self.auv_y + RELATIVE_Y
            self.modem_z = ABSOLUTE_Z
            rospy.loginfo("modem init at {:.1f} {:.1f} {:.1f}".format(
                self.modem_x, self.modem_y, self.modem_z
            ))
            self.has_nav = True


if __name__ == '__main__':
    node = SlantRange()
    rospy.spin()
