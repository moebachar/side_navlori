from setuptools import setup

package_name = 'wifi_scanner'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='navlori',
    maintainer_email='navlori@localhost',
    description='Publishes raw WiFi scans (sudo iw dev wlan0 scan) as JSON on /wifi/rssi',
    license='MIT',
    entry_points={
        'console_scripts': [
            'scanner = wifi_scanner.scanner:main',
        ],
    },
)
