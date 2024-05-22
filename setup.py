from setuptools import find_packages, setup

package_name = 'stretch_vlm_pkg'

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
    maintainer='sriram',
    maintainer_email='srirampr@umich.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'text_publisher = stretch_vlm_pkg.text_publisher:main',
            'py_trees_test = stretch_vlm_pkg.py_trees_test:main',
        ],
    },
)
