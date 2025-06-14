import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, Image
from sensor_msgs_py import point_cloud2
import cv2
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener
from mbes_pipe_detection.tf2_sensor_msgs import do_transform_cloud

from scipy.interpolate import griddata
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

        self.create_timer(0.1, self.pcl_patch_callback)
        self.pcl_patch_pub = self.create_publisher(
            PointCloud2,
            'pcl_patch_for_pipeline_detection',
            10
        )

        self.cv_bridge = CvBridge()
        self.create_timer(1.0 / self.detection_frequency, self.detection_callback)
        self.detection_image_pub = self.create_publisher(
            Image,
            'pipeline_detection_image',
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
        self.declare_parameter('detection_frequency', 1.)  # Hz
        self.detection_frequency = self.get_parameter('detection_frequency').get_parameter_value().double_value
        self.declare_parameter('normalize_intensity', True)
        self.normalize_intensity = self.get_parameter('normalize_intensity').get_parameter_value().bool_value
        self.declare_parameter('resolution', 0.5)  # meters
        self.resolution = self.get_parameter('resolution').get_parameter_value().double_value
        self.get_logger().info(f'Pipeline detection parameters: num_pings={self.num_pings_for_detection}, '
                               f'detection_frequency={self.detection_frequency}, '
                               f'normalize_intensity={self.normalize_intensity}, resolution={self.resolution}')

    def _initiate_circular_buffer(self, msg):
        """
        Initializes the circular buffer with the first point cloud message.
        """
        num_bins = msg.width
        self.get_logger().info(f'Number of bins in pcl: {num_bins}')
        self.circular_pcl_buffer = np.zeros((self.num_pings_for_detection, num_bins, 4), dtype=np.float32)
        self.get_logger().info(f'Initialized circular pcl buffer with shape: {self.circular_pcl_buffer.shape}')
        self.fields = msg.fields
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

        if self.circular_pcl_buffer is None:
            self._initiate_circular_buffer(msg=msg)

        self.update_circular_buffer(msg)

    def update_circular_buffer(self, msg):
        """
        Updates the circular buffer with the latest point cloud message.
        Ignores the message if all xyz values are zero.
        """

        pcl = point_cloud2.read_points_numpy(msg, ['x', 'y', 'z', 'intensity'])
        if pcl.size == 0 or np.all(pcl[:, :3] == 0):
            self.get_logger().warn('Received point cloud with all xyz values as zero, ignoring this ping.')
            return

        self.circular_pcl_buffer[self.ping_counter % self.num_pings_for_detection, ...] = pcl.reshape(-1, 4)
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

    def detection_callback(self):
        """
        This callback is called at the specified detection frequency.
        It retrieves the ordered pings from the circular buffer and performs pipeline detection.
        The results are published as an Image message.
        """
        intensity_image = self.construct_intensity_image_from_circular_pcl_buffer()

    def construct_intensity_image_from_circular_pcl_buffer(self):
        """
        Constructs an intensity image from the ordered pings.
        """
        ordered_pings = self.get_ordered_pings(normalize=self.normalize_intensity)
        if ordered_pings is None:
            return None

        x = ordered_pings[:, :, 0]
        y = ordered_pings[:, :, 1]
        z = ordered_pings[:, :, 2]
        intensities = ordered_pings[:, :, 3]

        x_min, x_max = (np.min(x), np.max(x))
        y_min, y_max = (np.min(y), np.max(y))
        num_x_pixels = int((x_max - x_min) / self.resolution)
        num_y_pixels = int((y_max - y_min) / self.resolution)
        self.get_logger().info(f'Constructing intensity image with shape: ({num_y_pixels}, {num_x_pixels})')
        X, Y = np.meshgrid(
            np.linspace(x_min, x_max, num_x_pixels),
            np.linspace(y_min, y_max, num_y_pixels)
        )
        intensity_image = griddata(
            (x.flatten(), y.flatten()),
            intensities.flatten(),
            (X, Y),
            method='linear',
        )
        intensity_image_normalized = cv2.normalize(
            intensity_image,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
        ).astype(np.uint8)
        image_msg = self.cv_bridge.cv2_to_imgmsg(intensity_image_normalized, encoding='mono8')
        self.detection_image_pub.publish(image_msg)


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