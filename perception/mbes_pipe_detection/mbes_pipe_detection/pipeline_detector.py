import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
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

        self.circular_pcl_buffer = None
        self.ping_counter = 0
        self.pcl_fields = None
        self.point_cloud_subscriber = self.create_subscription(
            PointCloud2,
            self.input_topic,
            self.point_cloud_callback,
            100
        )


        self.create_timer(1.0 / self.detection_frequency, self.pcl_patch_callback)
        self.pcl_patch_pub = self.create_publisher(
            PointCloud2,
            'pipeline_detection',
            10
        )


    def _declare_and_initialize_parameters(self):
        self.declare_parameter('input_topic', '/lolo/sensors/mbes/bathymetry/points')
        self.input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        self.declare_parameter('output_topic', 'pipeline_detection')
        self.output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        self.declare_parameter('frame_id', 'lolo/base_link')
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        self.declare_parameter('utm_zone', '33')
        self.utm_zone = self.get_parameter('utm_zone').get_parameter_value().string_value
        self.declare_parameter('utm_band', 'V')
        self.utm_band = self.get_parameter('utm_band').get_parameter_value().string_value
        self.utm_frame = f'utm_{self.utm_zone}_{self.utm_band}'
        self.get_logger().info(f'Using UTM frame: {self.utm_frame}')

        # Declare parameters for pipeline detection
        self.declare_parameter('num_pings_for_detection', 100)
        self.num_pings_for_detection = self.get_parameter('num_pings_for_detection').get_parameter_value().integer_value
        self.declare_parameter('detection_frequency', 1)  # Hz
        self.detection_frequency = self.get_parameter('detection_frequency').get_parameter_value().double_value
        self.declare_parameter('normalize_intensity', True)
        self.normalize_intensity = self.get_parameter('normalize_intensity').get_parameter_value().bool_value

    def _initiate_circular_buffer(self, pcl, fields):
        """
        Initializes the circular buffer with the first point cloud data.
        This function is called when the first point cloud message is received.
        """
        num_bins = pcl.shape[0]
        self.get_logger().info(f'Number of bins in pcl: {num_bins}')
        self.circular_pcl_buffer = np.zeros((self.num_pings_for_detection, num_bins, 4), dtype=np.float32)
        self.get_logger().info(f'Initialized circular pcl buffer with shape: {self.circular_pcl_buffer.shape}')
        self.fields = fields
        self.get_logger().info(f'Point cloud fields: {self.fields}')

    def point_cloud_callback(self, msg):
        """
        Callback function for the point cloud subscriber.
        This function processes incoming PointCloud2 messages, transforms them to the UTM frame if necessary,
        and appends the points to the circular_pcl_buffer.
        """
        # Transform the point cloud to the desired frame if necessary
        if msg.header.frame_id != self.utm_frame:
            try:
                transform = self.tf_buffer.lookup_transform(self.utm_frame, msg.header.frame_id, rclpy.time.Time())
                msg = do_transform_cloud(msg, transform)
            except Exception as e:
                self.get_logger().error(f'Error transforming point cloud: {e}')
                return

        pcl = point_cloud2.read_points_numpy(msg, ['x', 'y', 'z', 'intensity'])
        if self.circular_pcl_buffer is None:
            self._initiate_circular_buffer(pcl=pcl, fields=msg.fields)

        # Append the new point cloud to the circular buffer
        self.circular_pcl_buffer[self.ping_counter % self.num_pings_for_detection] = pcl.reshape(1, -1, 4)
        self.ping_counter += 1


    def get_ordered_pings(self, normalize=True):
        """
        Returns an ordered chronological view of the circular point cloud buffer.
        If not enough pings have been received, it returns None.
        """
        if self.ping_counter < self.num_pings_for_detection:
            return None
        start_index = self.ping_counter % self.num_pings_for_detection
        ordered_pings = np.roll(self.circular_pcl_buffer, -start_index, axis=0)

        if normalize:
            mean_intensity = np.mean(ordered_pings[:, :, 3], axis=0)
            ordered_pings[..., -1] /= mean_intensity
        return ordered_pings



    def pcl_patch_callback(self):
        """
        This callback is called at the specified detection frequency.
        It retrieves the ordered pings from the circular buffer and publish the point cloud
        patch in chronological order for pipeline detection.
        """
        ordered_pings = self.get_ordered_pings(normalize=self.normalize_intensity)
        if ordered_pings is None:
            self.get_logger().info('Not enough pings received yet for detection.')
            return

        header = Header()
        header.frame_id = self.utm_frame
        header.stamp = self.get_clock().now().to_msg()
        fields = self.fields
        points = point_cloud2.create_cloud(header, fields, ordered_pings.reshape(-1, 4))
        self.pcl_patch_pub.publish(points)


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