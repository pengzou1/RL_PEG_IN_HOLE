#!/usr/bin/python3
import URBasic
import time
import math
import NetFT
import numpy as np
import csv
s2tcptran = np.array([[0.5179, -0.8959, 0], [0.8959, 0.5179, 0],
                      [0, 0, 1]])  # sensor to tcp rotation matrix,expressed in sensor frame
G = np.array([0, 0, -11.1431])  # tool gravity,expressed in base frame
p = np.array([[0, -0.0545,  0.0034], [0.0545, 0, 0.0023],
              [- 0.0034, -0.0023, 0]])  # mass center in sensor frame
# sensor initial offset
fd = np.array([-0.3819, -1.3764, -8.9912,  -0.0131,    -0.0841,      -0.0328])
tcp2sensor = np.linalg.inv(s2tcptran)
zero = np.zeros((3, 3), dtype=float)


class ImpedanceController:
    def __init__(self):
        self.robot = URBasic.urScriptExt.UrScriptExt(
            host='192.168.1.50', robotModel=URBasic.robotModel.RobotModel())
        # print(self.robot.get_actual_tcp_pose()[3:6])
        self.sensor = NetFT.Sensor('192.168.1.30')
        self.M = np.diag((0.008, 0.008, 0.008, 0.008, 0.008, 0.008))
        self.B = np.diag((160, 160, 800, 16, 16, 16))
        self.v = [0, 0, 0, 0, 0, 0]
        self.vd = [0, 0, 0, 0, 0, 0]
        self.Tc = 0.008
        self.MNum = np.linalg.inv(self.M+self.B*self.Tc)
        self.target_ft = [0, 0, 15, 0, 0, 0]
        self.target_ft = np.reshape(self.target_ft, (6, 1))
        self.v = np.reshape(self.v, (6, 1))
        self.vd = np.reshape(self.vd, (6, 1))

    def AxisAng2RotaMatri(self, angle_vec):
        '''
        Convert an Axis angle to rotation matrix
        AxisAng2Matrix(angle_vec)for i in range(3):
                if abs(self.v[i]) < 0.05 or abs(self.v[i]) > 1.5:
                    self.v[i] = 0
            for i in range(3, 6):
                if abs(self.v[i]) < 0.01 or abs(self.v[i]) > 1.0:
                    self.v[i] = 0
        angle_vec need to be a 3D Axis angle
        '''
        theta = math.sqrt(angle_vec[0]**2+angle_vec[1]**2+angle_vec[2]**2)
        if theta == 0.:
            return np.identity(3, dtype=float)

        cs = np.cos(theta)
        si = np.sin(theta)
        e1 = angle_vec[0]/theta
        e2 = angle_vec[1]/theta
        e3 = angle_vec[2]/theta

        R = np.zeros((3, 3))
        R[0, 0] = (1-cs)*e1**2+cs
        R[0, 1] = (1-cs)*e1*e2-e3*si
        R[0, 2] = (1-cs)*e1*e3+e2*si
        R[1, 0] = (1-cs)*e1*e2+e3*si
        R[1, 1] = (1-cs)*e2**2+cs
        R[1, 2] = (1-cs)*e2*e3-e1*si
        R[2, 0] = (1-cs)*e1*e3-e2*si
        R[2, 1] = (1-cs)*e2*e3+e1*si
        R[2, 2] = (1-cs)*e3**2+cs
        return R

    def RotatMatr2AxisAng(self, Matrix):
        '''
        Convert the rotation matrix to axis angle
        '''
        R = Matrix
        theta = np.arccos(0.5*(R[0, 0]+R[1, 1]+R[2, 2]-1))
        e1 = (R[2, 1]-R[1, 2])/(2*np.sin(theta))
        e2 = (R[0, 2]-R[2, 0])/(2*np.sin(theta))
        e3 = (R[1, 0]-R[0, 1])/(2*np.sin(theta))
        axis_ang = np.array([theta*e1, theta*e2, theta*e3])
        return axis_ang

    def gravitycomp(self, ft, rotate_i):
        '''
         this is a gravit compensation function to compensate the tool gravity and obtain the precise
        force on the load.
        '''
        # axisangle = self.robot.get_target_tcp_poImpedancese()[3:6]
        # rotate = URBasic.kinematic.AxisAng2RotaMatri(axisangle)
        # tcp rotation matrix expressed in tcp frame
        trans = np.dot(s2tcptran, rotate_i)
        # in sensor frame
        Ftrans = np.dot(trans, G)
        Mtrans = np.dot(p, Ftrans)
        fttrans = np.hstack((Ftrans, Mtrans))
        ftcomp = ft-fttrans-fd
        # print('ok')
        return ftcomp

    def move2initpose(self):
        x = [-0.11953694869004786, -0.4636732413647742, 0.19956611419633097,
             0.34418941658334096, 3.122197481868604, -0.03536070824609871]

        self.robot.movel(pose=x, a=1.2, v=1.0)

    def set_pose_noise(self):
        init_pose = self.robot.get_actual_tcp_pose()
        axisangle = init_pose[-3:]
        init_rotate = self.AxisAng2RotaMatri(axisangle)
        theta = 4*math.pi/360
        Rx = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)],
                       [0, np.sin(theta), np.cos(theta)]])
        Ry = np.array([[np.cos(theta), 0, np.sin(theta)], [
                      0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
        Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
                       [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        rotate_noi = np.dot(init_rotate, Rx)
        pose_noi = init_pose[:]
        pose_noi[-3:] = self.RotatMatr2AxisAng(rotate_noi)
        # pose_noi[0] += 0.001
        self.robot.movel(pose=pose_noi.tolist(), a=1.2, v=1.0)
        print(pose_noi)

    def correct_bias(self):
        ft = np.array(self.sensor.tare()) / 1000000.0
        pose = self.robot.get_actual_tcp_pose()
        axisangle = pose[-3:]
        rotate = self.AxisAng2RotaMatri(axisangle)
        self.init_ft = self.gravitycomp(ft, np.linalg.inv(rotate))
        print(self.init_ft)

    def get_ftbase(self):
        ft = np.array(self.sensor.tare()) / 1000000.0
        pose = self.robot.get_actual_tcp_pose()
        axisangle = pose[-3:]
        rotate = self.AxisAng2RotaMatri(axisangle)
        ftcomp = self.gravitycomp(ft, np.linalg.inv(rotate))-self.init_ft
        base2sensor = np.dot(rotate, tcp2sensor)
        base2sensor1 = np.c_[base2sensor, zero]
        base2sensor2 = np.c_[zero, base2sensor]
        base2sensortran = np.r_[base2sensor1, base2sensor2]
        ft_base = np.reshape(np.dot(base2sensortran, ftcomp), (6, 1))
        return ft_base

    def imp_run(self, ft_base):
        err = ft_base-self.target_ft
        print(err)

        self.v = np.dot(self.MNum * self.Tc, err) + np.dot(self.MNum * self.M, self.vd)
        for i in range(2):
            if abs(err[i]) < 0.005 or abs(err[i]) > 150:
                self.v[i] = 0
        for i in range(3, 6):
            if abs(err[i]) < 0.001 or abs(err[i]) > 10:
                self.v[i] = 0
        for i in range(2):
            if abs(self.v[i]) < 0.5 or abs(self.v[i]) > 1.5:
                self.v[i] = 0
        for i in range(3, 6):
            if abs(self.v[i]) < 0.1 or abs(self.v[i]) > 1.0:
                self.v[i] = 0
        self.v[2] = (err[2]*0.002)/(1+math.exp(abs(ft_base[2])/30))
        self.v[3:6] = -self.v[3:6]
        # print(self.v)
        self.vd = self.v
        v_tmp = self.v.tolist()
        v_cmd = [i for item in v_tmp for i in item]
        print(v_cmd)
        self.robot.speedl(xd=v_cmd, wait=False, a=1.0)

    def comptest(self):
        # init_ft = np.array(self.sensor.tare()) / 1000000.0
        ft = np.array(self.sensor.tare()) / 1000000.0
        pose = self.robot.get_actual_tcp_pose()
        axisangle = pose[-3:]
        rotate = self.AxisAng2RotaMatri(axisangle)
        init_ft = self.gravitycomp(ft, np.linalg.inv(rotate))
        ft = np.array(self.sensor.tare()) / 1000000.0
        pose = self.robot.get_actual_tcp_pose()
        axisangle = pose[-3:]
        rotate = self.AxisAng2RotaMatri(axisangle)
        ftcomp = self. gravitycomp(ft, np.linalg.inv(rotate))-init_ft
        print(init_ft)


if __name__ == "__main__":
    controller = ImpedanceController()
    controller.move2initpose()
    controller.set_pose_noise()
    controller.correct_bias()
    initp = controller.robot.get_actual_tcp_pose()
    z0 = initp[2]
    while abs(controller.robot.get_actual_tcp_pose()[2]-z0) < 0.036:
        ft_base = controller.get_ftbase()
        controller.imp_run(ft_base)
    controller.robot.stopl()
    controller.robot.close()
    print('success')
    # controller.comptest()
