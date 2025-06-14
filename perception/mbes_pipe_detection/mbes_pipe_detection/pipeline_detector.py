import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
from sensor_msgs_py import point_cloud2
from tf2_ros import Buffer, TransformListener
from mbes_pipe_detection.tf2_sensor_msgs import do_transform_cloud

import numpy as np

class PipelineDetector(Node):
    def __init__(self):
        super().__init__('pipeline_detector')
        self.get_logger().info('Pipeline Detector Node has been started.')
        self._declare_and_initialize_parameters()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)


    def _declare_and_initialize_parameters(self):
        self.declare_parameter('input_topic', '/lolo/sensors/mbes/bathymetry/points')
        self.input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        self.declare_parameter('frame_id', 'lolo/base_link')
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        self.declare_parameter('utm_zone', '33')
        self.utm_zone = self.get_parameter('utm_zone').get_parameter_value().string_value
        self.declare_parameter('utm_band', 'V')
        self.utm_band = self.get_parameter('utm_band').get_parameter_value().string_value
        self.utm_frame = f'utm_{self.utm_zone}_{self.utm_band}'
        self.get_logger().info(f'Using UTM frame: {self.utm_frame}')

        self.declare_parameter('num_pings', 50)
        self.num_pings = self.get_parameter('num_pings').get_parameter_value().integer_value
        self.declare_parameter('detection_frequency', 0.1)  # Hz
        self.detection_frequency = self.get_parameter('detection_frequency').get_parameter_value().double_value

def main(args=None):
    rclpy.init(args=args)
    node = PipelineDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()