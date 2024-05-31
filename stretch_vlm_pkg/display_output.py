import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/inference_output',
            self.image_callback,
            10)
        self.bridge = CvBridge()

    def image_callback(self, msg):
        if msg is None:
            print('No image data received')
        else:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            # cv2.imshow('Inference Output', cv_image)
            # cv2.waitKey(1)
            cv2.imwrite('/home/sriram/vlm_ws/src/stretch_vlm_pkg/image_folder/test_image_01.jpg', cv_image)

def main(args=None):
    rclpy.init(args=args)

    image_subscriber = ImageSubscriber()

    while rclpy.ok():
        rclpy.spin_once(image_subscriber)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    image_subscriber.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()