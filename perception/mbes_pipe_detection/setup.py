from setuptools import find_packages, setup

package_name = 'mbes_pipe_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Li Ling',
    maintainer_email='liling@kth.se',
    description='Pipeline detection from MBES point clouds',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pipeline_detector = mbes_pipe_detection.pipeline_detector:main',
        ],
    },
)
