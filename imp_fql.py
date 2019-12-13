import FuzzySet
import StateVariable
import FQL
import FIS
from impedance_controller import ImpedanceController
import matplotlib.pyplot as plt


# Create FIS
x1 = StateVariable.InputStateVariable(FuzzySet.Trapeziums(-100, -10.5, -9.5, -5.5), FuzzySet.Triangles(-10, -5, 0),
                                      FuzzySet.Triangles(5, 0, 5), FuzzySet.Triangles(0, 5, 10), FuzzySet.Trapeziums(5.5, 9.5, 10.5, 100))
x2 = StateVariable.InputStateVariable(FuzzySet.Trapeziums(-100, -5.25, -4.75, -2.75), FuzzySet.Triangles(-5, -2.5, 0),
                                      FuzzySet.Triangles(-2.5, 0, 2.5), FuzzySet.Triangles(0, 2.5, 5), FuzzySet.Trapeziums(2.75, 4.75, 5.25, 100))
# x3 = StateVariable.InputStateVariable(FuzzySet.Trapeziums(-100, -5.25, -4.75, -2.75), FuzzySet.Triangles(-5, -2.5, 0), FuzzySet.Triangles(-2.5, 0, 2.5), FuzzySet.Triangles(0, 2.5, 5), FuzzySet.Trapeziums(2.75, 4.75, 5.25, 100))
fis = FIS.Build(x1, x2)


# Create Model
# position_list = []
vel_list = []
fh_list = []
action_list = []
reward_list = []
model = FQL.Model(gamma=0.7, alpha=0.5, ee_rate=0.001, past_weight=0.9, q_initial_value='zero',
                  action_set_length=3, fis=fis)
controller = ImpedanceController()
for episodes in range(0, 1000):
    # if iteration % 1000 == 0 and iteration <= 20000:
    #     env.__init__()
    #     action = model.get_initial_action(env.state)
    #     reward, state_value = env.apply_action(action)
    # action = model.run(state_value, reward)
    # reward, state_value = env.apply_action(action)
    controller.__init__()
    controller.move2initpose()
    controller.set_pose_noise()
    controller.correct_bias()
    while (controller.z < 0.036):
        ft_base = controller.get_ftbase()
        controller.imp_run(ft_base)
        state_value, reward = controller.get_state_reward('rx')
        action = model.run(state_value, reward)
        controller.apply_action('rx', action)
        # position_list.append(state_value[0])
        vel_list.append(state_value[1])
        fh_list.append(state_value[2])
        action_list.append(action)
        reward_list.append(reward)

plt.figure(1)
plt.plot(position_list)
plt.ylabel('position')


plt.figure(2)
plt.plot(vel_list)
plt.ylabel('vel')

plt.figure(3)
plt.plot(fh_list)
plt.ylabel('fh')

plt.figure(4)
plt.plot(action_list)
plt.ylabel('damping')

plt.figure(5)
plt.plot(reward_list)
plt.ylabel('reward')

plt.show()
