# ros-melodic-ronin-py
A ROS Melodic node for calculating 2D position data from IMU accelerometer and gyroscope values. It is based on [RoNIN](https://github.com/Sachini/ronin).
See the original work and paper here: [Yan, H., Herath, S. and Furukawa, Y. (2019). RoNIN: Robust Neural Inertial Navigation in the Wild: Benchmark, Evaluations, and New Methods. [online] arXiv.org. Available at: https://arxiv.org/abs/1905.12853](https://arxiv.org/abs/1905.12853)
**Paper**: https://arxiv.org/abs/1905.12853  
**Website**: http://ronin.cs.sfu.ca/  
**Demo**: https://youtu.be/JkL3O9jFYrE

---
### Requirements
ROS-Melodic, python3, numpy, numpy-quaternion, torch, tqdm, scipy

### Usage:
1. Clone the repository.
2. Make sure the ROS is [capable of using Python3](https://medium.com/@beta_b0t/how-to-setup-ros-with-python-3-44a69ca36674).
3. Download the pre-trained **ronin_resnet** model from [HERE](https://www.dropbox.com/sh/3adl8zyp2y91otf/AABIRBecKwMJotMSrWE0z2n0a?dl=0). 
4. The node resides in **ros_melodic_ronin** package. It subscribes to **ronin_imu** topic for sensor_msgs/Imu messages. It only considers accelerometer and gyroscope data.
   The output position is published in **ronin_odo** topic in nav_msgs/Odometry messages. Only the position X and Y values are updated.
5. Consult the launch files for model parameters. The contained values are the same as were used to train the model.

Resource usage is 15% of an Intel i7-7700 using the default parameters.
