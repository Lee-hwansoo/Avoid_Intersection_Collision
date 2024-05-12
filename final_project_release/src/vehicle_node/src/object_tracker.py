# coding: utf-8
import numpy as np
import time
from state_filters import Extended_KalmanFilter, IMM_filter
from state_models import CA, CTRA

class DynamicObstacleTracker:
    def __init__(self, dt=0.1, T=1):
        mat_trans = np.array([[0.85, 0.15], [0.15, 0.85]])
        mu = [1.0, 0.0]

        self.dt = dt
        self.T = T

        self.filters = [Extended_KalmanFilter(5, 4),
                        Extended_KalmanFilter(6, 4)]

        self.models = [CA(self.dt), CTRA(self.dt)]

        self.Q_list = [[0.1, 0.1, 0.1, 0.1, 0.001],
                       [0.1, 0.1, 0.1, 0.1, 0.001, 0.01]]

        for i in range(len(self.filters)):
            self.filters[i].F = self.models[i].step
            self.filters[i].H = self.models[i].H
            self.filters[i].JA = self.models[i].JA
            self.filters[i].JH = self.models[i].JH
            self.filters[i].Q = np.diag(self.Q_list[i])
            self.filters[i].R = np.diag([0.1, 0.1, 0.1, 0.1])

        self.IMM = IMM_filter(self.filters, mu, mat_trans)
        self.MM = [mu]
        self.X = []

    def initialize(self, data):
        # [x, y, v, a, theta], [x, y, v, a, theta, theta_rate]
        # x = [np.array([data[0], data[1], data[2], data[3], data[4]]), np.array([data[0], data[1], data[2], data[3], data[4], data[5]])]

        # temp data
        x = [np.array([data[0], data[1], data[3], 0, data[2]]),
                     np.array([data[0], data[1], data[3], 0, data[2], 0])]

        for i in range(len(self.filters)):
            self.filters[i].x = x[i]

        self.X.append(x[1])

    def update(self, data):
        # [x, y, v, theta]
        # z = [data[0], data[1], data[2], data[3]]

        # temp data
        z = [data[0], data[1],
                data[3], data[2]]

        self.IMM.prediction()
        self.IMM.merging(z)

        while len(self.MM) > int(self.T/self.dt):
            self.MM.pop(0)
        while len(self.X) > int(self.T/self.dt):
            self.X.pop(0)

        self.MM.append(self.IMM.mu.copy())
        self.X.append(self.IMM.x.copy())

    def predict(self):
        traj = self.IMM.predict(self.T)
        return traj

class MultiDynamicObstacleTracker:
    def __init__(self, dt=0.1, T=1, timeout=1.5):
        self.trackers = {}
        self.dt = dt
        self.T = T
        self.timeout = timeout  # 객체의 타임아웃 시간 (초)

    def add_tracker(self, obj_id):
        if obj_id not in self.trackers.keys():
            self.trackers[obj_id] = {
                'tracker': DynamicObstacleTracker(dt=self.dt, T=self.T),
                'last_update_time': time.time()  # 객체가 마지막으로 업데이트된 시간 초기화
            }

    def initialize(self, obj_id, data):
        if obj_id in self.trackers.keys():
            self.trackers[obj_id]['tracker'].initialize(data)
            self.trackers[obj_id]['last_update_time'] = time.time()  # 업데이트된 시간 갱신
        else:
            self.add_tracker(obj_id)
            self.trackers[obj_id]['tracker'].initialize(data)
            self.trackers[obj_id]['last_update_time'] = time.time()  # 업데이트된 시간 갱신

    def delete(self, obj_id):
        if obj_id in self.trackers.keys():
            del self.trackers[obj_id]
        else:
            print(f"obj_id")

    def update(self, obj_id, data):
        if obj_id in self.trackers.keys():
            current_time = time.time()  # 현재 시간
            if current_time - self.trackers[obj_id]['last_update_time'] > self.timeout:
                self.delete(obj_id)  # 타임아웃된 객체 삭제
                pass

            self.trackers[obj_id]['tracker'].update(data)
            self.trackers[obj_id]['last_update_time'] = time.time()  # 업데이트된 시간 갱신
        else:
            self.initialize(obj_id, data)

    def predict(self):
        trajs = {}
        for obj_id in self.trackers.keys():
            trajs[obj_id] = self.trackers[obj_id]['tracker'].predict()

        if len(trajs) == 0:
            return None
        else:
            return trajs

# if __name__ == "__main__":
#     # MultiDynamicObstacleTracker 객체 생성
#     tracker = MultiDynamicObstacleTracker(dt=0.1, T=10, timeout=1.5)

#     # 초기화할 객체 데이터 생성
#     obj_id = 1
#     initial_data = [10, 20, 5, 0, np.pi/4, 0.1]  # [x, y, v, a, theta, theta_rate]

#     # 초기화 수행
#     tracker.initialize(obj_id, initial_data)

#     # 업데이트할 데이터 생성
#     update_data = [12, 22, 6, np.pi/3]  # [x, y, v, theta]

#     # 데이터 업데이트 수행
#     tracker.update(obj_id, update_data)

#     # 예측 결과 얻기
#     prediction = tracker.predict()

#     # 예측 결과 출력
#     print("Predicted Trajectory:")
#     print(prediction)

# class DynamicObstacleTrackerNode:
#     def __init__(self):
#         rospy.init_node('dynamic_obstacle_tracker_node', anonymous=True)
#         rospy.Subscriber('object_info_topic', ObjectInfo, self.object_info_callback)
#         self.object_info_pub = rospy.Publisher('tracked_object_info_topic', ObjectInfo, queue_size=10)
#         self.object_path_pub = rospy.Publisher('tracked_object_path_topic', Path, queue_size=10)
#         self.tracker = MultiDynamicObstacleTracker()

#     def object_info_callback(self, msg):
#         obj_id = msg.id
#         data = [msg.x, msg.y, msg.v, msg.a, msg.theta, msg.theta_rate]

#         self.tracker.update(obj_id, data)
#         trajs = self.tracker.predict()

#     def run(self):
#         rospy.spin()

# if __name__ == "__main__":
#     try:
#         node = DynamicObstacleTrackerNode()
#         node.run()
#     except rospy.ROSInterruptException:
#         pass
