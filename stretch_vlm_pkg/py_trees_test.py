import py_trees
import py_trees_ros
from py_trees_ros import utilities as util
from py_trees.behaviour import Behaviour
from py_trees.common import Status, OneShotPolicy
from py_trees.decorators import OneShot
from py_trees import (
    display as display_tree,
    logging as log_tree
)
import rclpy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import String
from sensor_msgs.msg import Image
import numpy as np
import cv2
from cv_bridge import CvBridge

class ProcessData(Behaviour):
    def __init__(self, name="ProcessData"):
        super(ProcessData, self).__init__(name)
        self.blackboard = py_trees.blackboard.Blackboard()
        self.cv_bridge = CvBridge()

    def update(self):
        if hasattr(self.blackboard, 'text_message') and hasattr(self.blackboard, 'image'):
            self.saved_text = self.blackboard.text_message
            self.saved_image = self.cv_bridge.imgmsg_to_cv2(self.blackboard.image, "bgr8")
            self.logger.debug(f"Saved text: {self.saved_text}")
            self.logger.debug(f"Saved image shape: {self.saved_image.shape}")
            print("Got text and image")
            return Status.SUCCESS
        return Status.FAILURE

def create_behavior_tree():
    root = py_trees.composites.Sequence("Root", memory=True)

    # Create a QoS profile for the subscribers
    # qos_profile = QoSProfile(
    #     reliability=QoSReliabilityPolicy.RELIABLE,
    #     history=QoSHistoryPolicy.KEEP_LAST,
    #     depth=10
    # )

    text_to_blackboard = py_trees_ros.subscribers.ToBlackboard(
        name="TextToBlackboard",
        topic_name="/text_topic",
        topic_type=String,
        qos_profile=util.qos_profile_unlatched(),
        blackboard_variables={'text_message': 'data'}
    )

    image_to_blackboard = py_trees_ros.subscribers.ToBlackboard(
        name="ImageToBlackboard",
        topic_name="/camera/color/image_raw",
        topic_type=Image,
        qos_profile=util.qos_profile_unlatched(),
        blackboard_variables={'image': None}
    )

    process_data = ProcessData()

    decorated_root = OneShot(name="OneShotDecorator", child=root, policy=OneShotPolicy.ON_COMPLETION)

    root.add_children([text_to_blackboard, image_to_blackboard, process_data])
    return decorated_root

def main(args=None):
    rclpy.init(args=args)
    root = create_behavior_tree()
    tree = py_trees_ros.trees.BehaviourTree(
        root=root,
        unicode_tree_debug=True
    )
    try:
        tree.setup(timeout=15)
    except py_trees_ros.exceptions.TimedOutError as e:
        console.logerror(console.red + "failed to setup the tree, aborting [{}]".format(str(e)) + console.reset)
        tree.shutdown()
        rclpy.try_shutdown()
        sys.exit(1)
    except KeyboardInterrupt:
        # not a warning, nor error, usually a user-initiated shutdown
        console.logerror("tree setup interrupted")
        tree.shutdown()
        rclpy.try_shutdown()
        sys.exit(1)

    tree.tick_tock(period_ms=1000.0)

    try:
        rclpy.spin(tree.node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        tree.shutdown()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()
