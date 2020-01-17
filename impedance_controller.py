#!/usr/bin/python3
import joblib
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
vcc = 0.1
acc = 0.1
d = 0.001
mlp = joblib.load('mlp2.pkl')
mean = np.array([-0.56343318, - 0.3632879, - 0.28735675, 0.09976162])
std = np.array([0.66760705, 0.92386432, 0.15856688, 0.05147471])


class ImpedanceController:
    robot = URBasic.urScriptExt.UrScriptExt(
        host='192.168.1.50', robotModel=URBasic.robotModel.RobotModel())
    # print(self.robot.get_actual_tcp_pose()[3:6])
    sensor = NetFT.Sensor('192.168.1.30')
    def __init__(self):

        self.M = np.diag((0.008, 0.008, 0.008, 0.008, 0.008, 0.008))
        self.B = np.diag((100, 100, 800, 30, 30, 30))
        self.v = [0, 0, 0, 0, 0, 0]
        self.vd = [0, 0, 0, 0, 0, 0]
        self.a = [0, 0, 0, 0, 0, 0]
        self.Tc = 0.008

        self.target_ft = [0, 0, 15, 0, 0, 0]
        self.target_ft = np.reshape(self.target_ft, (6, 1))
        self.v = np.reshape(self.v, (6, 1))
        self.vd = np.reshape(self.vd, (6, 1))
        self.a = np.reshape(self.vd, (6, 1))
        self.z = 0
        self.lastz = 0
        self.dz = 0
        self.P = np.diag((0.00, 0.00, 0.0001, 0.00, 0.00, 0.00))
        self.I = np.diag((0, 0, 0.0001, 0, 0, 0))
        self.D = np.diag((0, 0, 0.0004, 0, 0, 0))
        self.move = np.array([[0, d, 0, 0, 0, 0], [0, -d, 0, 0, 0, 0],
                              [-d, 0, 0, 0, 0, 0], [d, 0, 0, 0, 0, 0]])

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

    def get_zdz(self):
        self.z = abs(self.robot.get_actual_tcp_pose()[2]-self.initz)
        self.dz = self.z-self.lastz
        self.lastz = self.z
        return self.z, self.dz

    def fuzzy_reward(self, f, m, z, dz):
        mfoutput = self.fuzzy_mf(f, m)
        zdzoutput = self.fuzzy_zdz(z, dz)
        reward = self.fuzzy_2(mfoutput, zdzoutput)
        return reward

    def fuzzy_2(self, mf, zdz):
        mf = -mf
        zdz = -zdz
        secnum = 4
        mf_max = 1
        everysecmf = mf_max / secnum
        secmf1 = everysecmf
        secmf2 = 2 * everysecmf
        secmf3 = 3 * everysecmf

        mf1 = 1 - mf / everysecmf
        mf2 = 1 - abs(mf - secmf1) / everysecmf
        mf3 = 1 - abs(mf - secmf2) / everysecmf
        mf4 = 1 - abs(mf - secmf3) / everysecmf
        mf5 = (mf - secmf3) / everysecmf

        zdz_max = 1
        everyseczdz = zdz_max / secnum
        seczdz1 = everyseczdz
        seczdz2 = 2 * everyseczdz
        seczdz3 = 3 * everyseczdz

        zdz1 = 1 - zdz / everyseczdz
        zdz2 = 1 - abs(zdz - seczdz1) / everyseczdz
        zdz3 = 1 - abs(zdz - seczdz2) / everyseczdz
        zdz4 = 1 - abs(zdz - seczdz3) / everyseczdz
        zdz5 = (zdz - seczdz3) / everyseczdz

        mf1 = max(min(mf1, 1.0), 0.0)
        mf2 = max(min(mf2, 1.0), 0.0)
        mf3 = max(min(mf3, 1.0), 0.0)
        mf4 = max(min(mf4, 1.0), 0.0)
        mf5 = max(min(mf5, 1.0), 0.0)

        zdz1 = max(min(zdz1, 1.0), 0.0)
        zdz2 = max(min(zdz2, 1.0), 0.0)
        zdz3 = max(min(zdz3, 1.0), 0.0)
        zdz4 = max(min(zdz4, 1.0), 0.0)
        zdz5 = max(min(zdz5, 1.0), 0.0)

        r1 = min(mf1, zdz1)
        r2 = min(mf1, zdz2)
        r3 = min(mf1, zdz3)
        r4 = min(mf1, zdz4)
        r5 = min(mf1, zdz5)
        r6 = min(mf2, zdz1)
        r7 = min(mf2, zdz2)
        r8 = min(mf2, zdz3)
        r9 = min(mf2, zdz4)
        r10 = min(mf2, zdz5)
        r11 = min(mf3, zdz1)
        r12 = min(mf3, zdz2)
        r13 = min(mf3, zdz3)
        r14 = min(mf3, zdz4)
        r15 = min(mf3, zdz5)
        r16 = min(mf4, zdz1)
        r17 = min(mf4, zdz2)
        r18 = min(mf4, zdz3)
        r19 = min(mf4, zdz4)
        r20 = min(mf4, zdz5)
        r21 = min(mf5, zdz1)
        r22 = min(mf5, zdz2)
        r23 = min(mf5, zdz3)
        r24 = min(mf5, zdz4)
        r25 = min(mf5, zdz5)

        r0 = r1+r2+r3+r4+r5+r6+r7+r8+r9+r10+r11+r12+r13+r14+r15+r16+r17+r18+r19+r20+r21+r22+r23+r24+r25

        output = (0.33*(r1+r6)+0.48*(r2+r7+r11+r16)+0.69*(r3+r12+r17+r21)
                  + 1*(r4+r8+r9+r13+r22)+1.44*(r5+r10+r14+r18+r23)+2.07*(r15+r19+r24)+3*(r20+r25))/(-3*r0)
        return output

    def fuzzy_zdz(self, z, dz):
        z = z*1000
        dz = dz*1000
        secnum = 4
        dz_max = 3
        everysecdz = dz_max / secnum
        secdz1 = everysecdz
        secdz2 = 2 * everysecdz
        secdz3 = 3 * everysecdz

        z_max = 26
        everysecz = z_max / secnum
        secz1 = everysecz
        secz2 = 2 * everysecz
        secz3 = 3 * everysecz

        z1 = 1 - z / everysecz
        z2 = 1 - abs((z - secz1) / everysecz)
        z3 = 1 - abs((z - secz2) / everysecz)
        z4 = 1 - abs((z - secz3) / everysecz)
        z5 = (z - secz3) / everysecz

        z1 = max(min(z1, 1.0), 0.0)
        z2 = max(z2, 0.0)
        z3 = max(z3, 0.0)
        z4 = max(z4, 0.0)
        z5 = max(min(z5, 1.0), 0.0)

        dz1 = 1 - dz / everysecdz
        dz2 = 1 - abs((dz - secdz1) / everysecdz)
        dz3 = 1 - abs((dz - secdz2) / everysecdz)
        dz4 = 1 - abs((dz - secdz3) / everysecdz)
        dz5 = (dz - secdz3) / everysecdz

        dz1 = max(min(dz1, 1.0), 0.0)
        dz2 = max(dz2, 0.0)
        dz3 = max(dz3, 0.0)
        dz4 = max(dz4, 0.0)
        dz5 = max(min(dz5, 1.0), 0.0)

        r1 = min(z1, dz1)
        r2 = min(z1, dz2)
        r3 = min(z1, dz3)
        r4 = min(z1, dz4)
        r5 = min(z1, dz5)
        r6 = min(z2, dz1)
        r7 = min(z2, dz2)
        r8 = min(z2, dz3)
        r9 = min(z2, dz4)
        r10 = min(z2, dz5)
        r11 = min(z3, dz1)
        r12 = min(z3, dz2)
        r13 = min(z3, dz3)
        r14 = min(z3, dz4)
        r15 = min(z3, dz5)
        r16 = min(z4, dz1)
        r17 = min(z4, dz2)
        r18 = min(z4, dz3)
        r19 = min(z4, dz4)
        r20 = min(z4, dz5)
        r21 = min(z5, dz1)
        r22 = min(z5, dz2)
        r23 = min(z5, dz3)
        r24 = min(z5, dz4)
        r25 = min(z5, dz5)
        r0 = r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8 + r9 + r10 + r11 + r12 + r13 + \
            r14 + r15 + r16 + r17 + r18 + r19 + r20 + r21 + r22 + r23 + r24 + r25

        # Outputs are 9 sections
        ratio = math.sqrt(2)
        zdzoutput = (4.0 * (r1 + r2) + 2.0 * ratio * (r6) + 2.0 * (r3 + r7 + r11) + ratio * (r8 + r9 + r12) +
                     1.0 * (r4 + r13) + 1 / ratio * (r5 + r16) + 0.5 * (r10 + r14 +
                                                                        r17 + r18) + 0.5 * ratio * (r15 + r19 + r21 + r22 + r23)
                     + 0.25 * (r20 + r24 + r25)) / -(4.0 * r0)
        # print(zdzoutput)
        return zdzoutput

    def fuzzy_mf(self, f, m):
        secnum = 4
        moment_max = 5
        everysecm = moment_max / secnum
        secm1 = everysecm
        secm2 = 2 * everysecm
        secm3 = 3 * everysecm

        force_max = 44
        everysecf = force_max / secnum
        secf1 = everysecf
        secf2 = 2 * everysecf
        secf3 = 3 * everysecf

        m1 = (m - secm3) / everysecm
        m2 = 1 - abs(m - secm3) / everysecm
        m3 = 1 - abs(m - secm2) / everysecm
        m4 = 1 - abs(m - secm1) / everysecm
        m5 = 1 - (m - secm1) / everysecm

        m1 = max(min(m1, 1.0), 0.0)
        m2 = max(m2, 0.0)
        m3 = max(m3, 0.0)
        m4 = max(m4, 0.0)
        m5 = max(min(m5, 1.0), 0.0)

        f1 = (f - secf3) / everysecf
        f2 = 1 - abs(f - secf3) / everysecf
        f3 = 1 - abs(f - secf2) / everysecf
        f4 = 1 - abs(f - secf1) / everysecf
        f5 = 1 - (f - secf1) / everysecf

        f1 = max(min(f1, 1.0), 0.0)
        f2 = max(f2, 0.0)
        f3 = max(f3, 0.0)
        f4 = max(f4, 0.0)
        f5 = max(min(f5, 1.0), 0.0)

        # Fuzzy rules
        r1 = min(m1, f1)
        r2 = min(m1, f2)
        r3 = min(m1, f3)
        r4 = min(m1, f4)
        r5 = min(m1, f5)
        r6 = min(m2, f1)
        r7 = min(m2, f2)
        r8 = min(m2, f3)
        r9 = min(m2, f4)
        r10 = min(m2, f5)
        r11 = min(m3, f1)
        r12 = min(m3, f2)
        r13 = min(m3, f3)
        r14 = min(m3, f4)
        r15 = min(m3, f5)
        r16 = min(m4, f1)
        r17 = min(m4, f2)
        r18 = min(m4, f3)
        r19 = min(m4, f4)
        r20 = min(m4, f5)
        r21 = min(m5, f1)
        r22 = min(m5, f2)
        r23 = min(m5, f3)
        r24 = min(m5, f4)
        r25 = min(m5, f5)

        # Outputs are 9 sections
        ratio = math.sqrt(2)
        r0 = r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8 + r9 + r10 + r11 + r12 + r13 + \
            r14 + r15 + r16 + r17 + r18 + r19 + r20 + r21 + r22 + r23 + r24 + r25
        mfoutput = (0.25 * (r1 + r2 + r6) + 0.5 * ratio * (r3 + r7 + r8) + 0.5 * (r4 + r9 + r11) + 1 / ratio * (r5 + r10 + r12) +
                    1.0 * (r13 + r16) + ratio * (r14 + r17 + r21 + r22) +
                    2.0 * (r15 + r18) + 2 * ratio * (r19 + r20 + r23)
                    + 4.0 * (r24 + r25)) / -(4 * r0)

        return mfoutput

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
        # x = [-0.11819919972034086, -0.45426283327612405, 0.1999228367933996, -
        #  0.31333842972576104, -3.1016449048271983, 0.0017124526375461655]
        self.robot.movel(pose=x, a=0.1, v=0.5)

    def set_pose_noise(self):
        init_pose = self.robot.get_actual_tcp_pose()
        axisangle = init_pose[-3:]
        init_rotate = self.AxisAng2RotaMatri(axisangle)
        theta = 3*math.pi/360
        Rx = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)],
                       [0, np.sin(theta), np.cos(theta)]])
        Ry = np.array([[np.cos(theta), 0, np.sin(theta)], [
                      0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
        Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
                       [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        rotate_noi = np.dot(init_rotate, Rx)
        pose_noi = init_pose[:]
        pose_noi[-3:] = self.RotatMatr2AxisAng(rotate_noi)
        pose_noi[0] += 0.003
        pose_noi[1] -= 0.003
        # pose_noi[0] += 0.001
        self.robot.movel(pose=pose_noi.tolist(), a=1.2, v=1.0)
        self.initz = self.robot.get_actual_tcp_pose()[2]
        # print(pose_noi)

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
        # print(self.B[3, 3])
        self.MNum = np.linalg.inv(self.M+self.B*self.Tc)
        err = ft_base-self.target_ft
        # print(err)

        self.v = np.dot(self.MNum * self.Tc, err) + np.dot(self.MNum * self.M, self.vd)
        for i in range(2):
            if abs(err[i]) < 0.005 or abs(err[i]) > 100:
                self.v[i] = 0
        for i in range(3, 6):
            if abs(err[i]) < 0.0001 or abs(err[i]) > 9:
                self.v[i] = 0
        # print(self.v)
        for i in range(3):
            if abs(self.v[i]) < 0.5 or abs(self.v[i]) > 1.5:
                self.v[i] = 0.000
                # self.vd[i] = 0

        for i in range(3, 6):
            if abs(self.v[i]) < 0.0001 or abs(self.v[i]) > 1.0:
                self.v[i] = 0.0
                # self.vd[i] = 0
        self.v[2] = (err[2]*0.002)/(1+math.exp(abs(ft_base[2])/30))
        self.v[3:6] = -self.v[3:6]
        # print(self.v)
        self.a = self.v-self.vd
        self.vd = self.v
        v_tmp = self.v.tolist()
        v_cmd = [i for item in v_tmp for i in item]
        # print(v_cmd)
        self.robot.speedl(xd=v_cmd, wait=False, a=1.0)
        ft_tmp = ft_base.tolist()
        ft_record = [i for item in ft_tmp for i in item]
        self.save2csv('/home/zp/github/RL_PEG_IN_HOLE/data2/imp_ft_fql5.csv', ft_record)
        self.save2csv('/home/zp/github/RL_PEG_IN_HOLE/data2/imp_v_fql2.csv', v_cmd)

    def save2csv(self, filepath, data):
        with open(filepath, 'a', newline='') as t:
            writer1 = csv.writer(t)
            writer1.writerow(data)

    def get_state_reward(self, variable_name):
        state = []
        ft = self.get_ftbase()
        if variable_name == 'x':
            state.append(self.v[0, 0])
            # res.append(self.a[0, 0])
            state.append(ft[0, 0])
            z, dz = self.get_zdz()
            reward = self.fuzzy_reward(abs(ft[0, 0]), abs(ft[4, 0]), z, dz)
            return state, reward
        elif variable_name == 'y':
            state.append(self.v[1, 0])
            # res.append(self.a[1, 0])
            state.append(ft[1, 0])
            z, dz = self.get_zdz()
            reward = self.fuzzy_reward(max(abs(ft[1, 0]), abs(ft[2, 0])), abs(ft[3, 0]), z, dz)
            return state, reward
        elif variable_name == 'rx':
            state.append(abs(self.v[3, 0]))
            # res.append(self.a[3, 0])
            state.append(abs(ft[3, 0]))
            # state.append(self.a[3, 0])
            z, dz = self.get_zdz()
            reward = self.fuzzy_reward(abs(ft[1, 0]), abs(ft[3, 0]), z, dz)
            # f = math.sqrt(ft[0, 0]*ft[0, 0]+ft[1, 0]*ft[1, 0]+ft[2, 0]*ft[2, 0])
            # m = math.sqrt(ft[3, 0]*ft[3, 0]+ft[4, 0]*ft[4, 0]+ft[5, 0]*ft[5, 0])
            # reward = self.fuzzy_reward(f, m, z, dz)
            return state, reward
        elif variable_name == 'ry':
            state.append(self.v[4, 0])
            # res.append(self.a[4, 0])
            state.append(ft[4, 0])
            z, dz = self.get_zdz()
            reward = self.fuzzy_reward(abs(ft[0, 0]), abs(ft[4, 0]), z, dz)
            return state, reward
        # elif variable_name == 'rz':
        #     res.append(self.v[5, 0])
        #     # res.append(self.a[5, 0])
        #     res.append(ft[5, 0])
        #     z, dz = self.get_zdz()
        #     reward = self.fuzzy_reward(ft[5, 0], ft[3, 0], z, dz)
        #     res.append(reward)

    def apply_action(self, variable_name, u):
        if variable_name == 'x':
            self.B[0, 0] = u
        elif variable_name == "y":
            self.B[1, 1] = u
        elif variable_name == "rx":
            self.B[3, 3] = u
        elif variable_name == "ry":
            self.B[4, 4] = u
        # elif variable_name == "rz":
            # self.B[5, 5] = u

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
        print(ftcomp)

    def search_hole(self):
        z0 = self.robot.get_actual_tcp_pose()[2]
        print('seh')
        while True:
            # if count > 1000:
            #     break
            currentPose = self.robot.get_actual_tcp_pose()
            pose_record = currentPose.tolist()
            with open("searchdata9.csv", 'a', newline='') as t:
                writer1 = csv.writer(t)
                writer1.writerow(pose_record)
            err_i = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            err_i = np.reshape(err_i, (6, 1))
            err_d = np.array([0, 0, 0, 0, 0, 0])
            err_d = np.reshape(err_d, (6, 1))
            for i in range(10):
                if self.robot.get_actual_tcp_pose()[2]-z0 < -0.005:
                    self.v = self.vd
                    print('suc')
                    self.robot.stopl()
                    #  self.robot.close()
                    return
                ft = np.array(self.sensor.tare()) / 1000000.0
                # print(ft)
                pose = self.robot.get_actual_tcp_pose()
                axisangle = pose[-3:]
                rotate = self.AxisAng2RotaMatri(axisangle)
                # 2.4926138340994397, -1.911217721409695, -0.006238534241991061]
                ftcomp = self.gravitycomp(ft, np.linalg.inv(rotate))-self.init_ft
                # ft_sensor_record = ftcomp.tolist()
                base2sensor = np.dot(rotate, tcp2sensor)
                base2sensor1 = np.c_[base2sensor, zero]
                base2sensor2 = np.c_[zero, base2sensor]
                base2sensortran = np.r_[base2sensor1, base2sensor2]
                # print(base2sensortran)
                ft_base = np.reshape(np.dot(base2sensortran, ftcomp), (6, 1))
                err = self.target_ft-ft_base
                # print(err)
                err_i = err_i + err
                err_l = err-err_d
                err_d = err
                # print(err)
                if abs(err[2]) < 1:
                    self.robot.stopl()
                    x = np.array([ft_base[0][0], ft_base[1][0], ft_base[3][0], ft_base[4][0]])
                    x_in = (x-mean)/std
                    nextdirection = mlp.predict(np.array([x_in]))
                    print(nextdirection)
                    nextPose = currentPose+self.move[nextdirection[0]]
                    # #print(nextPose)
                    self.robot.movel(pose=nextPose.tolist(), a=acc, v=vcc)
                    break
                self.v = -(np.dot(self.P, err)+np.dot(self.D, err_l) +
                           np.dot(self.I, err_i))/(1+math.exp(abs(ft_base[2])/30))
                v_tmp = self.v.tolist()
                v_cmd = [i for item in v_tmp for i in item]
                self.robot.speedl(xd=v_cmd, wait=False, a=1.0)
                ft_tmp = ft_base.tolist()
                ft_record = [i for item in ft_tmp for i in item]
                self.save2csv('/home/zp/github/RL_PEG_IN_HOLE/data2/imp_ft_fql5.csv', ft_record)


if __name__ == "__main__":
    time.sleep(5)
    controller = ImpedanceController()
    controller.move2initpose()
    controller.set_pose_noise()
    controller.correct_bias()
    controller.comptest()
    # initp = controller.robot.get_actual_tcp_pose()
    # z0 = initp[2]
    # # time.sleep(2.0)
    # while abs(controller.robot.get_actual_tcp_pose()[2]-z0) < 0.036:
    #     ft_base = controller.get_ftbase()
    #     controller.imp_run(ft_base)
    # controller.robot.stopl()
    # controller.robot.close()
    # print('success')

    # controller.move2initpose()
    # controller.set_pose_noise()
    controller.search_hole()
    z = 0
    while z < 0.04:
        ft_base = controller.get_ftbase()
        controller.imp_run(ft_base)
        z, dz = controller.get_zdz()
    #         controller.save2csv('/home/zp/github/RL_PEG_IN_HOLE/data/zdzexp.csv', [z, dz])
    # controller.comptest()

    controller.robot.stopl()
    controller.robot.close()
    print('success')
