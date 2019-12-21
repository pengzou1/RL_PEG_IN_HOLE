#!/usr/bin/python3
import FuzzySet
import StateVariable
import FQL
import FIS
import time
from impedance_controller import ImpedanceController
import matplotlib.pyplot as plt

ft_max = 5
v_max = 0.08
sec = 4
ft_sec = ft_max/sec
v_sec = v_max/sec
# Create FIS
x2 = StateVariable.InputStateVariable(FuzzySet.Triangles(-ft_sec, 0, ft_sec), FuzzySet.Triangles(0, ft_sec, 2*ft_sec),
                                      FuzzySet.Triangles(ft_sec, 2*ft_sec, 3*ft_sec), FuzzySet.Triangles(2*ft_sec, 3*ft_sec, 4*ft_sec), FuzzySet.Trapeziums(3*ft_sec, 4*ft_sec, 8*ft_sec, 12*ft_sec))
x1 = StateVariable.InputStateVariable(FuzzySet.Triangles(-v_sec, 0, v_sec), FuzzySet.Triangles(0, v_sec, 2*v_sec),
                                      FuzzySet.Triangles(v_sec, 2*v_sec, 3*v_sec), FuzzySet.Triangles(2*v_sec, 3*v_sec, 4*v_sec), FuzzySet.Trapeziums(3*v_sec, 4*v_sec, 8*v_sec, 12*v_sec))
# x3 = StateVariable.InputStateVariable(FuzzySet.Trapeziums(-100, -5.25, -4.75, -2.75), FuzzySet.Triangles(-5, -2.5, 0), FuzzySet.Triangles(-2.5, 0, 2.5), FuzzySet.Triangles(0, 2.5, 5), FuzzySet.Trapeziums(2.75, 4.75, 5.25, 100))
# x3 = StateVariable.InputStateVariable(FuzzySet.Triangles(-0.15, -0.1, -0.05), FuzzySet.Triangles(-0.1, -0.05, 0.0),
# FuzzySet.Triangles(-0.05, 0.0, 0.05), FuzzySet.Triangles(0.0, 0.05, 0.1), FuzzySet.Triangles(0.05, 0.1, 0.15))
fis = FIS.Build(x1, x2)


# Create Model
# position_list = []

model = FQL.Model(gamma=0.1, alpha=0.15, ee_rate=1.0, past_weight=0.9, q_initial_value='file',
                  action_set_length=3, fis=fis)
controller = ImpedanceController()
controller.move2initpose()
controller.set_pose_noise()
controller.correct_bias()
step_max = 500
for episodes in range(1, 6):
    # if iteration % 1000 == 0 and iteration <= 20000:
    #     env.__init__()
    #     action = model.get_initial_action(env.state)
    #     reward, state_value = env.apply_action(action)
    # action = model.run(state_value, reward)
    # reward, state_value = env.apply_action(action)
    # filepath = '/home/zp/github/RL_PEG_IN_HOLE/data/fqltest.csv'
    controller.__init__()
    controller.move2initpose()
    controller.set_pose_noise()
    # controller.correct_bias()
    # state_init, reward = controller.get_state_reward('rx')
    # # state_init[0] = max(min(state_init[0], 0.09), -0.029)
    # # state_init[1] = max(min(state_init[0], 2.9), -5.9)
    # # state_init = [state_init[0], state_init[1]]
    # action = model.get_initial_action(state_init)
    # controller.apply_action('rx', action)
    for step in range(1, step_max):
        if controller.z > 0.026:
            break
        data_record = []
        state_value, reward = controller.get_state_reward('rx')
        action = model.test(state_value)
        print(action)
        controller.apply_action('rx', action)
        ft_base = controller.get_ftbase()
        controller.imp_run(ft_base)
        ft_tmp = ft_base.tolist()
        ft_record = [i for item in ft_tmp for i in item]
        controller.save2csv('/home/zp/github/RL_PEG_IN_HOLE/data//fql/fqltest_ft' +
                            str(episodes)+'.csv', ft_record)
        data_record.append(state_value[0])
        data_record.append(state_value[1])
        data_record.append(action)
        controller.save2csv('/home/zp/github/RL_PEG_IN_HOLE/data//fql/fqltest_act' +
                            str(episodes)+'.csv', data_record)
        # state_value[0] = max(min(state_value[0], 0.09), -0.029)
        # state_value[1] = max(min(state_value[1], 2.9), -5.9)
        # state_value = [state_value[0], state_value[1]]
        # if controller.z > 0.026:
        #     reward = reward+1-step/step_max
        #     action = model.run(state_value, reward)  # update q table
        #     data_record.append(state_value[0])
        #     data_record.append(state_value[1])
        #     # data_record.append(state_value[2])
        #     data_record.append(action)
        #     data_record.append(reward)
        #     controller.save2csv(filepath, data_record)
        #     model.save_qtable()
        #     break
        # else:


controller.robot.stopl()
controller.robot.close()
# position_list.append(state_value[0])
# vel_list.append(state_value[0])
# fh_list.append(state_value[1])
# action_list.append(action)
# reward_list.append(reward)

# plt.figure(1)
# plt.plot(position_list)
# plt.ylabel('position')
#
#
# plt.figure(2)
# plt.plot(vel_list)
# plt.ylabel('vel')
#
# plt.figure(3)
# plt.plot(fh_list)
# plt.ylabel('fh')
#
# plt.figure(4)
# plt.plot(action_list)
# plt.ylabel('damping')
#
# plt.figure(5)
# plt.plot(reward_list)
# plt.ylabel('reward')
#
# plt.show()
