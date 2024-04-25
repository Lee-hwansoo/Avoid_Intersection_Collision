#!/bin/bash
echo '프로젝트 설정...'

source /opt/ros/noetic/setup.bash

catkin_make

echo 'source /root/final_project/devel/setup.bash' >> ~/.bashrc

pip3 install IPython scipy

echo '프로젝트 설정이 완료되었습니다.'
