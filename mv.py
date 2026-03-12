import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

def save_animation_2link(z_array, l1, l2, actions, reward_peak_step,DT, filename="two_link_qlearn.mp4"):

    theta1 = z_array[:,0]
    theta1_dot = z_array[:,1]
    theta2 = z_array[:,2]
    theta2_dot = z_array[:,3]

    # 第1リンク先端
    x1 = l1 * np.cos(theta1)
    y1 = l1 * np.sin(theta1)

    # 第2リンク先端（手先）
    x2 = x1 + l2 * np.cos(theta1 + theta2)
    y2 = y1 + l2 * np.sin(theta1 + theta2)

    fig, (ax_anim, ax_plot) = plt.subplots(2,1,figsize=(8,10))

    # アニメーション描画範囲
    max_range = l1 + l2 + 0.5
    ax_anim.set_xlim(-max_range, max_range)
    ax_anim.set_ylim(-max_range, max_range)
    ax_anim.set_aspect('equal')

    # 動くリンク描画
    link1_line, = ax_anim.plot([], [], "bo-", lw=3)
    link2_line, = ax_anim.plot([], [], "ro-", lw=3)

    # ピーク姿勢の固定リンク
    peak_x1 = x1[reward_peak_step]
    peak_y1 = y1[reward_peak_step]
    peak_x2 = x2[reward_peak_step]
    peak_y2 = y2[reward_peak_step]

    peak_link1_line, = ax_anim.plot([0, peak_x1], [0, peak_y1], "yo-", lw=3)
    peak_link2_line, = ax_anim.plot([peak_x1, peak_x2], [peak_y1, peak_y2], "yo-", lw=3)


    
    # 状態量プロット
    ax_left = ax_plot
    ax_right = ax_plot.twinx()

    line_theta1, = ax_left.plot([], [], label="theta1", color="blue")
    line_theta2, = ax_left.plot([], [], label="theta2", color="green")
    line_theta1_dot, = ax_right.plot([], [], label="theta1_dot", color="red")
    line_theta2_dot, = ax_right.plot([], [], label="theta2_dot", color="purple")
    line_action, = ax_left.plot([], [], label="action", color="orange")

    lines = [line_theta1, line_theta2, line_theta1_dot, line_theta2_dot, line_action]
    labels = [l.get_label() for l in lines]

    ax_plot.legend(lines,labels,loc="upper right")

    def init():
        link1_line.set_data([], [])
        link2_line.set_data([], [])
        line_theta1.set_data([], [])
        line_theta2.set_data([], [])
        line_theta1_dot.set_data([], [])
        line_theta2_dot.set_data([], [])
        line_action.set_data([], [])
        return link1_line, link2_line, peak_link1_line, peak_link2_line

    def update(i):
        time=np.arange(i)*DT

        # アニメーション部分
        link1_line.set_data([0, x1[i]], [0, y1[i]])
        link2_line.set_data([x1[i], x2[i]], [y1[i], y2[i]])

        # ピークフレームに文字を表示
        if i==reward_peak_step:
            ax_anim.text(0.1,0.9,f"Reward peak={reward_peak_step}",transform=ax_anim.transAxes,color="red", fontsize=16, fontweight="bold")

        # 状態量
        line_theta1.set_data(time, theta1[:i])
        line_theta2.set_data(time, theta2[:i])
        line_theta1_dot.set_data(time, theta1_dot[:i])
        line_theta2_dot.set_data(time, theta2_dot[:i])
        line_action.set_data(time, actions[:i])

        ax_plot.set_xlim(0, len(z_array))
        ax_plot.relim()
        ax_plot.autoscale_view()
        ax_right.relim()
        ax_right.autoscale_view()

        return link1_line, link2_line, peak_link1_line, peak_link2_line

    ani = animation.FuncAnimation(
        fig, update, frames=len(z_array),
        init_func=init, interval=1, blit=False
    )

    ani.save("/home/tamaki/2link_mv.mp4", writer="ffmpeg", fps=100)
    plt.close()
