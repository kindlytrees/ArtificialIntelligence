# Software Preparation

## 前置软件安装

- 配置 WSLg 或 手动安装 X Server,WSL里输入

```bash
# powershell 管理员权限执行 wsl --update
sudo apt update
sudo apt install x11-apps
```

## RO2（humble）安装

- 添加源

```bash
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

echo "添加源完成"
```

- 安装ros2  
```bash
#安装ROS2
sudo apt update -y
sudo apt upgrade -y
sudo apt install ros-humble-desktop
sudo apt install ros-dev-tools
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

- 安装Gazebo等模拟器工具

```bash
#ROS 2 Humble	Gazebo Fortress (Gazebo Sim 7)	官方推荐。Humble 开始转向新一代 Gazebo Sim。
sudo apt install ros-humble-turtlebot3-simulations
sudo apt-get install gz-fortress
```

