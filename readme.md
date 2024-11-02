# VAFR 2024 LAB 1 GROUP 6
Tested on Ubunut 22.05 using ROS2 Humble
## Instalation
### RAE-ROS
Requirese [RAE-ROS](https://github.com/luxonis/rae-ros) msgs package, which can be installed using the following commands:
```
mkdir git
cd git
git clone https://github.com/luxonis/rae-ros.git
cd rae-ros
MAKEFLAGS="-j1 -l1" colcon build --symlink-install --packages-select rae_msgs
source ./install/setup.bash
```
rae-ros may require some of these dependecies to be installed:
```
sudo apt install libgpiod-dev
sudo apt install libmpg123-dev
pip install ffmpeg
sudo apt install libsndfile1-dev

```
### LAB 1 Package
Make sure you are in the LAb1_WS folder
```
colcon build --packages-select lab1
source ./install/setup.bash
```

## Running:
### Edge detector
```
ros2 run lab1 edge_detector
```

 
