#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import copy
import bisect
import matplotlib.cm as cm
import matplotlib.animation as animation

from IPython.display import HTML
from utils import *
from agent import agent


import tf
import rospkg
import rospy

from geometry_msgs.msg import Twist, Point32, PolygonStamped, Polygon, Vector3, Pose, Quaternion, Point
from visualization_msgs.msg import MarkerArray, Marker

from std_msgs.msg import Float32, Float64, Header, ColorRGBA, UInt8, String, Float32MultiArray, Int32MultiArray

# 시뮬레이션 환경을 관리하는 클래스
class Environments(object):
    def __init__(self, course_idx, dt=0.1, min_num_agent=8):
        # 생성자: 초기 상태 설정
        self.spawn_id = 0 # 생성될 에이전트의 ID를 0에서 시작
        self.vehicles = {} # 에이전트들을 저장할 딕셔너리
        self.int_pt_list = {} # 각 에이전트별 교차점 정보를 저장할 딕셔너리
        self.min_num_agent = min_num_agent # 환경 내에서 유지되어야하는 최소 에이전트 수
        self.dt = dt # 시뮬레이션의 시간 단계 (초)
        self.course_idx = course_idx # 사용할 코스의 인덱스

        self.object_data = {}  # 객체별 센서 데이터 이력 저장
        self.object_recent_count = {}  # 각 객체의 최근 관측 카운트
        self.recent_count_threshold = 2  # 객체 데이터 삭제 전 허용되는 최대 미관측 횟수
        self.max_data_points = 5  # 각 객체별로 유지할 최대 데이터 포인트 수, 0.1초 간격으로 5개

        self.window_size = 2  # 이동 평균 필터의 윈도우 크기

        self.initialize() # 초기화 함수 호출

    # 환경 초기화 함수
    def initialize(self, init_num=6):
        # 일시 정지 상태를 False로 설정
        self.pause = False
        # ROS 파라미터 서버에서 파일 경로 읽기
        filepath = rospy.get_param("file_path")
        Filelist = glob.glob(filepath+"/*info.pickle")

        file = Filelist[0]

        with open(file, "rb") as f:
            Data = pickle.load(f)

        # 데이터에서 맵 포인트와 연결 정보 추출
        self.map_pt = Data["Map"]
        self.connectivity = Data["AS"]

        # 지정된 초기 에이전트 수만큼 에이전트 생성
        for i in range(init_num):
            if i==0:
                # 첫 번째 에이전트는 특정 코스에 생성
                CourseList = [[4,1,18], [4,2,25], [4,0,11]]
                self.spawn_agent(target_path = CourseList[self.course_idx], init_v = 0)
            else:
                # 나머지는 무작위 위치에 생성
                self.spawn_agent()

    # 새 에이전트를 생성하는 함수
    def spawn_agent(self, target_path=[], init_v = None):
        # 초기 점유 상태는 True로 설정
        is_occupied = True

        if target_path:
            # target_path가 지정된 경우, 초기 점유 상태를 False로 설정
            spawn_cand_lane = target_path[0]
            is_occupied = False
            s_st = 5 # 시작 s 좌표 설정
        else:
            # target_path가 지정되지 않은 경우, 무작위 레인 선택
            spawn_cand_lane = [10,12,24,17,19]
            s_st = np.random.randint(0,20)
            max_cnt = 10
            while(is_occupied and max_cnt>0):
                spawn_lane = np.random.choice(spawn_cand_lane)
                is_occupied = False
                for id_ in self.vehicles.keys():
                    if (self.vehicles[id_].lane_st == spawn_lane) and np.abs(self.vehicles[id_].s - s_st) < 25:
                        is_occupied = True
                max_cnt-=1

        # 점유되지 않은 경우 에이전트 생성
        if is_occupied is False:
            if target_path:
                target_path = target_path
            else:
                target_path = [spawn_lane]
                spawn_lane_cand = np.where(self.connectivity[spawn_lane]==1)[0]
                while(len(spawn_lane_cand)>0):
                    spawn_lane = np.random.choice(spawn_lane_cand)
                    target_path.append(spawn_lane)
                    spawn_lane_cand = np.where(self.connectivity[spawn_lane]==1)[0]

            # 연결된 레인을 통해 에이전트의 경로 설정
            target_pt = np.concatenate([self.map_pt[lane_id][:-1,:] for lane_id in target_path], axis=0)
            self.int_pt_list[self.spawn_id] = {}

            # 기존의 모든 에이전트와의 교차점을 찾음
            for key in self.vehicles.keys():
                intersections = find_intersections(target_pt[:,:3], self.vehicles[key].target_pt[:,:3]) # ((x,y), i, j)
                if intersections:
                    self.int_pt_list[self.spawn_id][key] = [(inter, xy[0], xy[1]) for (inter, xy) in intersections]
                    self.int_pt_list[key][self.spawn_id] = [(inter, xy[1], xy[0]) for (inter, xy) in intersections]

            # 정지선과 종료선 인덱스 설정
            stopline_idx = len(self.map_pt[target_path[0]])-1
            endline_idx = len(self.map_pt[target_path[0]])+len(self.map_pt[target_path[1]])-2

            # 에이전트 객체 생성 및 초기화
            self.vehicles[self.spawn_id] = agent(self.spawn_id, target_path, s_st, target_pt, dt=self.dt, init_v = init_v,
                                                 stoplineidx = stopline_idx, endlineidx = endline_idx)
            self.spawn_id +=1 # 에이전트 ID 증가

    # 에이전트를 삭제하는 함수
    def delete_agent(self):
        delete_agent_list = []
        for id_ in self.vehicles.keys():
            if (self.vehicles[id_].target_s[-1]-10) < self.vehicles[id_].s:
                delete_agent_list.append(id_)

        return delete_agent_list

    # 상대 좌표를 절대 좌표로 변환하는 함수
    def relative_to_absolute(self, sdv, sensor_info):
        # SDV의 현재 위치와 헤딩 가져오기
        sdv_x = sdv.x
        sdv_y = sdv.y
        sdv_h = sdv.h

        absolute_sensor_info = []
        for info in sensor_info:
            obj_id, rel_x, rel_y, rel_h, rel_vx, rel_vy = info

            # 회전 변환을 사용해 상대 위치를 절대 위치로 변환
            cos_h = np.cos(sdv_h)
            sin_h = np.sin(sdv_h)
            abs_x = sdv_x + cos_h * rel_x - sin_h * rel_y
            abs_y = sdv_y + sin_h * rel_x + cos_h * rel_y

            # 객체와 SDV 위치 사이의 절대 각도 계산
            angle_to_obj = np.arctan2(abs_y - sdv_y, abs_x - sdv_x) - sdv_h
            # 각도를 -π에서 π 사이로 정규화
            angle_to_obj = np.arctan2(np.sin(angle_to_obj), np.cos(angle_to_obj))

            # 변환된 데이터를 리스트에 추가
            absolute_sensor_info.append([obj_id, abs_x, abs_y, rel_h, rel_vx, rel_vy, angle_to_obj])

        return absolute_sensor_info

    # 필터링된 센서 데이터를 반환하는 함수
    def filter_sensor_data(self, obj_id, new_data):
        # 이동 평균 필터 적용 로직
        all_data = self.object_data[obj_id] + [new_data]
        x_values = [d[1] for d in all_data]  # 데이터의 x 좌표
        y_values = [d[2] for d in all_data]  # 데이터의 y 좌표
        h_values = [d[3] for d in all_data]  # 데이터의 heading 차이

        if len(all_data) < self.window_size:
            filtered_x = moving_average(x_values, len(all_data))
            filtered_y = moving_average(y_values, len(all_data))
            filtered_h = moving_average(h_values, len(all_data))
        else:
            filtered_x = moving_average(x_values, self.window_size)
            filtered_y = moving_average(y_values, self.window_size)
            filtered_h = moving_average(h_values, self.window_size)

        filtered_data = [obj_id, filtered_x[-1], filtered_y[-1], filtered_h[-1]] + new_data[4:]
        return filtered_data

    # 센서 데이터 업데이트 함수
    def update_sensor_data(self, absolute_sensor_info):
        current_detected = set()  # 이번 업데이트에서 감지된 객체 ID들 저장

        # 첫 번째 관측 이후 필터링 적용하여 센서 데이터 업데이트
        for data in absolute_sensor_info:
            # data = [obj_id, abs_x, abs_y, rel_h, rel_vx, rel_vy]
            obj_id = data[0]
            current_detected.add(obj_id)
            if obj_id not in self.object_data:
                self.object_data[obj_id] = [data]  # 최초 감지 시 데이터 리스트 생성
                self.object_recent_count[obj_id] = 1  # 최초 감지 시 카운트 1로 초기화
            else:
                if len(self.object_data[obj_id]) >= 1:  # 첫 관측 이후 데이터에 대해 필터링 적용
                    filtered_data = self.filter_sensor_data(obj_id, data)
                    self.object_data[obj_id].append(filtered_data)  # 기존 데이터 리스트에 추가
                else: # 첫 관측 데이터는 그대로 추가
                    self.object_data[obj_id].append(data)

                # 데이터 관리
                self.object_recent_count[obj_id] = min(self.recent_count_threshold, self.object_recent_count[obj_id] + 1)  # 카운트 증가하지만 최대값 초과 금지
                # 데이터 리스트가 최대 크기를 초과하면 가장 오래된 데이터 제거
                if len(self.object_data[obj_id]) > self.max_data_points:
                    self.object_data[obj_id].pop(0)


        # 미감지된 객체 카운트 감소 및 필요시 삭제
        for obj_id in list(self.object_recent_count.keys()):
            if obj_id not in current_detected:
                self.object_data[obj_id].pop(0)
                self.object_recent_count[obj_id] -= 1
                if self.object_recent_count[obj_id] == 0:
                    del self.object_data[obj_id]
                    del self.object_recent_count[obj_id]

    # 시뮬레이션 실행 함수
    def run(self):
        for id_ in self.vehicles.keys():
            if id_ == 0:
                # 첫 번째 에이전트에 대한 센서 정보와 로컬 경로 정보를 얻음
                sensor_info = self.vehicles[id_].get_measure(self.vehicles) # [ obj id, rel x, rel y, rel h, rel vx, rel vy ]
                local_lane_info = self.vehicles[id_].get_local_path()
                """
                To Do

                """
                absolute_sensor_info = self.relative_to_absolute(self.vehicles[id_], sensor_info) # [ obj id, abs x, abs y, rel h, rel vx, rel vy, angle_to_obj ]
                # print("상대적", sensor_info)
                # print("절대적", absolute_sensor_info)

                self.update_sensor_data(absolute_sensor_info)

                # print("----------\nData count per object ID:")
                # for obj_id in self.object_data.keys():
                #     data_list = self.object_data[obj_id]
                #     print(f"object Id {obj_id}, len : {len(data_list)}\n")
                #     # 가장 최신 데이터, 즉 가장 마지막 데이터를 활용
                #     print(f"Latest data: {data_list[-1]}\n")
                # print("----------")

                deceleration_distance = 8
                distance_to_nearest = float('inf')

                # 전방 각도 사이의 차량과의 가장 가까운 거리 계산
                for obj_id in self.object_data.keys():
                    latest_data = self.object_data[obj_id][-1]
                    # print(f"object ID {obj_id}의 각도 : {latest_data[6]}")
                    if latest_data[6] <= 28 * (np.pi / 180) and latest_data[6] >= -28 * (np.pi / 180):
                        distance = np.sqrt((self.vehicles[id_].x - latest_data[1])**2 + (self.vehicles[id_].y - latest_data[2])**2)
                        if distance < distance_to_nearest:
                            distance_to_nearest = distance

                if distance_to_nearest < deceleration_distance:
                    print("Break")
                    self.vehicles[id_].step_manual(ax=-3.5, steer=0)
                else:
                    print("Go")
                    self.vehicles[id_].step_manual(ax=0.5, steer=0)

            if id_  > 0 :
                # 나머지 에이전트에 대해 자동 제어 단계를 실행
                self.vehicles[id_].step_auto(self.vehicles, self.int_pt_list[id_])

    # 에이전트 재생성 함수
    def respawn(self):
        # 환경 내의 에이전트 수가 최소 수보다 적은 경우 에이전트를 추가 생성
        if len(self.vehicles)<self.min_num_agent:
            self.spawn_agent()

if __name__ == '__main__':
    try:
        f = Environments()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start node.')
