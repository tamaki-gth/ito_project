import numpy as np
import matplotlib.pyplot as plt

def plot_state(z_history,actions,max_step,tau):
    theta1=z_history[:,0]
    theta1_dot=z_history[:,1]
    theta2=z_history[:,2]
    theta2_dot=z_history[:,3]
    steps=np.arange(len(theta1))
    time=tau*steps

    fig,ax1=plt.subplots(figsize=(12,8))
    ax2=ax1.twinx()

    line_t1,=ax1.plot(time,theta1,label="theta1",color="blue")
    line_t2,=ax1.plot(time,theta2,label="theta2",color="green")
    line_act,=ax1.plot(time,actions,label="action",color="orange",alpha=0.5)

    line_t1d,=ax2.plot(time,theta1_dot,label="theta1_dot",color="red")
    line_t2d,=ax2.plot(time,theta2_dot,label="theta2_dot",color="purple")

    ax1.axvline(max_step*tau,color="black",linestyle="--",linewidth=2)
    ax1.set_title("State variables (best episode)")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Angle / Action")
    ax2.set_ylabel("Angular velocity")

    lines=[line_t1,line_t2,line_act,line_t1d,line_t2d]
    labels=[l.get_label() for l in lines]
    ax1.legend(lines,labels,loc="upper right")

    plt.tight_layout()
    plt.show()


def plot_rewards(R_history,ce_step_history,v2_history,R_distance_history,max_step,tau):
    R=np.array(R_history)
    ce_step=np.array(ce_step_history)
    ce_total=np.cumsum(ce_step)
    v2=np.array(v2_history)
    R_dis=np.array(R_distance_history)
    steps=np.arange(len(R))
    time=tau*steps

    fig,ax1=plt.subplots(figsize=(12,8))
    ax2=ax1.twinx()

    line_R,=ax1.plot(time,R,label="Total reward",color="blue")
    line_v2,=ax1.plot(time,v2,label="Hand speed",color="green")
    line_R_dis,=ax1.plot(time,R_dis,label="Flying distance reward",color="c")
    
    ax1.axvline(max_step*tau,color="black",linestyle="--",linewidth=2)
    ax1.text(max_step*tau,R.max(),f"Peak step={max_step}",color="black",fontsize=12,fontweight="bold")

    line_ce_step,=ax2.plot(time,ce_step,label="ce_step",color="red")
    line_ce_total,=ax2.plot(time,ce_total,label="ce_total",color="orange")

    ax1.set_title("Reward breakdown (best episode)")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Flying distance reward / Hand speed")
    ax2.set_ylabel("Energy")

    lines=[line_R,line_v2,line_R_dis,line_ce_step,line_ce_total]
    labels=[l.get_label() for l in lines]
    ax1.legend(lines,labels,loc="upper left")

    plt.tight_layout()
    plt.show()


def plot_energy(z_history,max_step,m1,m2,l1,l2,p1,p2,J1,J2,g,tau):
    theta1=z_history[:,0]
    theta1_dot=z_history[:,1]
    theta2=z_history[:,2]
    theta2_dot=z_history[:,3]

    T1=0.5*m1*(p1*theta1_dot)**2+0.5*J1*theta1_dot**2
    T2=0.5*m2*((l1*theta1_dot)**2+(p2*theta2_dot)**2+2*l1*p2*theta1_dot*theta2_dot*np.cos(theta2))+0.5*J2*(theta1_dot+theta2_dot)**2

    V1=m1*g*p1*np.sin(theta1)
    V2=m2*g*(l1*np.sin(theta1)+p2*np.sin(theta1+theta2))

    steps=np.arange(len(theta1))
    time=tau*steps

    fig,ax1=plt.subplots(figsize=(12,8))
    ax2=ax1.twinx()

    line_T1,=ax1.plot(time,T1,label="Kinetic_1",color="blue")
    line_T2,=ax1.plot(time,T2,label="Kinetic_2",color="green")

    line_V1,=ax2.plot(time,V1,label="Potential_1",color="red")
    line_V2,=ax2.plot(time,V2,label="Potential_2",color="orange")

    ax1.axvline(max_step*tau,color="black",linestyle="--",linewidth=2)

    ax1.set_title("Energy")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Kinetic energy")
    ax2.set_ylabel("Potential energy")
        
    lines=[line_T1,line_T2,line_V1,line_V2]
    labels=[l.get_label() for l in lines]
    ax1.legend(lines,labels,loc="upper right")

    plt.tight_layout()
    plt.show()
    
                   
    

    
    

    
