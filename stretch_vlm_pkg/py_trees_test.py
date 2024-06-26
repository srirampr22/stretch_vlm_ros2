import py_trees
import py_trees_ros
from py_trees_ros import utilities as util
from py_trees.behaviour import Behaviour
from py_trees.common import Status, OneShotPolicy
from py_trees.decorators import OneShot
# from py_trees.visitors import TreeToMsgVisitor
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

from groundingdino.util.inference import load_model, load_image, predict, annotate
from groundingdino.util.inference import Model as vlm_model

class ProcessData(Behaviour):
    def __init__(self, name="ProcessData"):
        super(ProcessData, self).__init__(name)
        self.blackboard = py_trees.blackboard.Blackboard()
        self.cv_bridge = CvBridge()
        self.bool_data = False
        self.bool_precidtion = False
        self.pre_processed_image = None
        self.TEXT_PROMPT = None
        self.BOX_TRESHOLD = 0.35
        self.TEXT_TRESHOLD = 0.25
        self.model = None
        self.model_config_path = "/home/sriram/vlm_ws/src/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        self.model_checkpoint_path = "/home/sriram/vlm_ws/src/GroundingDINO/weights/groundingdino_swint_ogc.pth"

    def setup(self, **kwargs):
        self.logger.info(f"{self.name}.setup()")
        try:
            self.node = kwargs['node']
        except KeyError as e:
            error_message = f"didn't find 'node' in setup's kwargs [{self.qualified_name}]"
            raise KeyError(error_message) from e

        self.model = load_model(
            model_config_path="/home/sriram/vlm_ws/src/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            model_checkpoint_path="/home/sriram/vlm_ws/src/GroundingDINO/weights/groundingdino_swint_ogc.pth"
        )
        self.logger.info("Model loaded")

    def initialise(self):
        self.logger.info(f"{self.name} [initialise]")
        if self.blackboard.exists('text_message') and self.blackboard.exists('image'):
            self.logger.info("Data received")
            # input_text = self.blackboard.text_message # only this line is throwing an error: AttributeError: 'Blackboard' object has no attribute 'text_message'
            input_text = self.blackboard.get('text_message')
            input_image = self.blackboard.get('image')
            cv2_image = self.cv_bridge.imgmsg_to_cv2(input_image, "bgr8") # image format is bgr8
            self.TEXT_PROMPT = input_text
            self.INPUT_IMAGE_CV2 = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
            
            self.logger.info(f"Text message: {self.TEXT_PROMPT}")

            self.pre_processed_image = vlm_model.preprocess_image(cv2_image)
            self.logger.info("Image preprocessed")

    def update(self):
        self.logger.info(f"{self.name} [update]")
        if self.pre_processed_image is not None:
            boxes, logits, phrases = predict(
                model=self.model,
                image=self.pre_processed_image,
                caption=self.TEXT_PROMPT,
                box_threshold=self.BOX_TRESHOLD,
                text_threshold=self.TEXT_TRESHOLD
            )

            if boxes is not None and logits is not None and phrases is not None:
                self.bool_precidtion = True
                # if boxes
            if self.bool_precidtion:
                self.logger.info(f"Inference successful, found {phrases}")
                post_processed_image = self.postprocessing(boxes, logits, phrases) # image is of the format bgr
                image_msg = self.cv2_to_image(post_processed_image)
                self.blackboard.set('inference_output', image_msg)  
                return Status.SUCCESS
            else:
                return Status.RUNNING

        return Status.FAILURE

    def terminate(self, new_status):
        self.logger.debug(f"Action::terminate {self.name} to {new_status}")
        
    def postprocessing(self, boxes, logits, phrases):
        image_source = self.INPUT_IMAGE_CV2
        self.logger.info(f"inference image shape: {image_source.shape}")
        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        return annotated_frame
    
    def cv2_to_image(self, cv2_image):
        image_msg = self.cv_bridge.cv2_to_imgmsg(cv2_image, encoding="bgr8")
        return image_msg

        
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
    
    wait_for_inf_output = py_trees.behaviours.WaitForBlackboardVariable(
        name="WaitForData",
        variable_name="/inference_output"    
        )
    
    inference_publisher = py_trees_ros.publishers.FromBlackboard(
        name="InferencePublisher",
        topic_name="/inference_output",
        topic_type=Image,
        qos_profile=util.qos_profile_unlatched(),
        blackboard_variable="/inference_output"
    )

    process_data = ProcessData()

    decorated_root = OneShot(name="OneShotDecorator", child=root, policy=OneShotPolicy.ON_COMPLETION)

    root.add_children([text_to_blackboard, image_to_blackboard, process_data, wait_for_inf_output, inference_publisher])
    return decorated_root

def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node('detection_tree')
    
    # Create the behavior tree
    root = create_behavior_tree()
    tree = py_trees_ros.trees.BehaviourTree(
        root=root,
        unicode_tree_debug=False
    )

    # Add visitors
    setup_logger = py_trees_ros.visitors.SetupLogger(node=node)
    tree_to_msg_visitor = py_trees_ros.visitors.TreeToMsgVisitor()
    # blackboard_visitor = BlackboardSnapshotVisitor()
    
    try:
        tree.setup(node=node, timeout=15, visitor=setup_logger)
    except py_trees_ros.exceptions.TimedOutError as e:
        py_trees.console.logerror(f"failed to setup the tree, aborting [{str(e)}]")
        tree.shutdown()
        rclpy.try_shutdown()
        sys.exit(1)
    except KeyboardInterrupt:
        py_trees.console.logerror("tree setup interrupted")
        tree.shutdown()
        rclpy.try_shutdown()
        sys.exit(1)

    node.get_logger().info("Starting behavior tree")
    
    try:
        while rclpy.ok():
            tree.tick()
            tree.visitors.append(tree_to_msg_visitor)
            # tree.visitors.append(blackboard_visitor)
            rclpy.spin_once(node, timeout_sec=0.1)

            # blackboard_snapshot = blackboard_visitor.visited
            if tree.root.status == py_trees.common.Status.FAILURE:
                node.get_logger().info("Behavior tree root returned FAILURE. Stopping...")
                break
    except KeyboardInterrupt:
        pass
    finally:
        tree.shutdown()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
