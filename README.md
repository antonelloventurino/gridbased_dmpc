# Grid-based Distributed Model Predictive Control for Differential Drive/Unicycle Robots
A ROS and Gazebo simulation package for distributed control of TurtleBot3 robots in industrial environments.

![](demo.gif)

## External Dependencies:
- numpy
- cvxpy
- scipy

Install Python dependencies with:
```
pip3 install -r requirements.txt
```

## System Requirements:
- Ubuntu 20.04 + ROS Noetic (recommended, tested stable)

## Build Instructions:
```
cd ~/catkin_ws
catkin_make --only-pkg-with-deps gridbased_dmpc
```

## Configuration:
Use this command to open _.bashrc_ file:
- ``` gedit ~/.bashrc ```

Make sure the following code exists in your `~/.bashrc` file or export them in terminal:
```
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
```
Add the _models_ directory to Gazebo's model path (only required once):
```
rospack profile
PACKAGE_PATH=$(rospack find gridbased_dmpc)
echo "export GAZEBO_MODEL_PATH=\$GAZEBO_MODEL_PATH:$PACKAGE_PATH/models" >> ~/.bashrc
source ~/.bashrc
```

## Running the Simulation:
The following command launches a simulation with three TurtleBot3 robots navigating a 3D industrial environment using distributed model predictive control:
```
roslaunch gridbased_dmpc main_multi.launch
```

## Citing this Work:
If you find this software useful for your research or development, please cite the following works:
- A. Venturino, L. Filice, G. Mezzatesta, F. Tedesco and G. Franzè, "Coordination of fleets of autonomous vehicles for logistics operations in industrial environments: a grid based receding horizon control approach," TechRxiv, December 10, 2024, doi: [10.36227/techrxiv.173386516.60845320/v1](https://doi.org/10.36227/techrxiv.173386516.60845320/v1).
```
@article{venturino2024coordination,
    author = {Antonello Venturino and Luigino Filice and Giovanni Mezzatesta and Francesco Tedesco and Giuseppe Franzè},
    title = {Coordination of fleets of autonomous vehicles for logistics operations in industrial environments: a grid based receding horizon control approach},
    journal = {TechRxiv},
    year = {2024},
    month = {December},
    doi = {10.36227/techrxiv.173386516.60845320/v1}
}
```
- A. Venturino, L. Filice and G. Franzè, "Grid-Based Receding Horizon Control for Unicycle Robots Under Logistic Operations," 2024 IEEE 29th International Conference on Emerging Technologies and Factory Automation (ETFA), Padova, Italy, 2024, pp. 1-7, doi: [10.1109/ETFA61755.2024.10710915](https://doi.org/10.1109/ETFA61755.2024.10710915).
```
@INPROCEEDINGS{venturino2024grid,
    author={Venturino, Antonello and Filice, Luigino and Franzè, Giuseppe},
    booktitle={2024 IEEE 29th International Conference on Emerging Technologies and Factory Automation (ETFA)},
    title={Grid-Based Receding Horizon Control for Unicycle Robots Under Logistic Operations},
    year={2024},
    volume={},
    number={},
    pages={1-7},
    doi={10.1109/ETFA61755.2024.10710915}
}
```

## License:
This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.

## Contact Information:
[Antonello Venturino](http://antonelloventurino.me/en/contact-me/)\
Emails:
- posta (at) antonelloventurino (.) me 
- antonello (.) venturino (at) unical (.) it
