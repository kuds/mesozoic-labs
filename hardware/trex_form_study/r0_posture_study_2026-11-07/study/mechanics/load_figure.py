"""Draw support geometry and hip-roll demand directly from solved poses."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import mujoco as mj
from screen import BASE,SIDES,fresh,home_command,set_pose,sole_corners

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('model',type=Path)
    p.add_argument('--out',type=Path,default=BASE/'single_support.png')
    a=p.parse_args();rows=json.loads((BASE/'metrics.json').read_text())
    model=a.model.stem;v=rows[model];m=mj.MjModel.from_xml_path(str(a.model));d=fresh(m)
    q=home_command(m);idx={m.joint(j).name:i for i,j in enumerate(m.actuator_trnid[:,0])}
    roll=v['single_support']['left']['hip_roll_offset_rad']
    for side in SIDES:
        q[idx[side+'_hip_roll']]+=roll;q[idx[side+'_ankle_roll']]-=roll
    q[idx['right_hip_pitch']]-=.25;q[idx['right_knee']]+=.5;q[idx['right_ankle_pitch']]-=.25
    set_pose(m,d,q,'left')
    fig,axs=plt.subplots(1,2,figsize=(11,4.8),gridspec_kw=dict(width_ratios=[1,1.35]),facecolor='#FCFBF7')
    ax,bx=axs
    for s,color in zip(SIDES,['#266457','#9d8069']):
        points=np.array([d.xanchor[m.joint(s+'_'+j).id,1:] for j in ['hip_roll','hip_pitch','knee','ankle_pitch','ankle_roll']])*1000
        cor=sole_corners(m,d,s)*1000
        ax.plot(*points.T,marker='o',color=color,lw=2.7,ms=4)
        ax.plot([cor[:,1].min(),cor[:,1].max()],[cor[:,2].mean()]*2,color=color,lw=6,solid_capstyle='butt')
    com=d.subtree_com[0]*1000
    ax.scatter(com[1],com[2],c='#b34630',s=50,zorder=3)
    ax.annotate('',xy=(com[1],3),xytext=(com[1],com[2]-8),arrowprops=dict(arrowstyle='-|>',color='#b34630',lw=1.3,ls='--'))
    ax.annotate('Whole-body COM',(com[1],com[2]),xytext=(-50,9),textcoords='offset points',color='#9e3825',fontsize=9)
    hip=d.xanchor[m.joint('left_hip_roll').id]*1000
    ax.plot([com[1],hip[1]],[hip[2]]*2,c='#607573',lw=.8,ls=':')
    ax.axhline(0,c='#49615e',lw=.8)
    ax.set(xlabel='Lateral position, mm  (+Y left)',ylabel='Height above floor, mm',aspect='equal',ylim=(-14,hip[2]+58))
    ax.set_title(f'{model}  /  left support',loc='left',weight='bold')
    labels=[k for k in rows if k.startswith(('A','B','C')) and k[-2:].isdigit()]
    baseline=[k for k in rows if k not in labels]
    labels=baseline+labels
    torques=[max(abs(rows[k]['single_support'][s]['joint_Nm'][s+'_hip_roll']) for s in SIDES) for k in labels]
    colors=['#777d77' if k in baseline else {'A':'#629690','B':'#607746','C':'#a06e44'}[k[0]] for k in labels]
    y=np.arange(len(labels))
    bx.barh(y,torques,color=colors,height=.64)
    bx.set_yticks(y,labels)
    bx.invert_yaxis();bx.set_xlim(0,max(torques)*1.15)
    for i,value in enumerate(torques):
        bx.text(value+.1,i,f'{value:.2f}',va='center',fontsize=8)
    bx.axvline(6,c='#992e20',ls='--',lw=1.5)
    bx.set(xlabel='Stance hip roll, N·m  (worse side)')
    bx.set_title('Prescribed one-foot gravity hold',loc='left',weight='bold')
    bx.legend(handles=[Line2D([0],[0],ls='--',color='#992e20',label='6 N·m sustained assumption')],loc='lower left',bbox_to_anchor=(0,-.26),frameon=False,fontsize=9)
    for panel in axs:
        panel.set_facecolor('#FCFBF7');panel.spines[['top','right']].set_visible(False)
        panel.spines[['bottom','left']].set_color('#86918a')
        panel.tick_params(labelsize=8,color='#86918a')
    fig.subplots_adjust(left=.08,right=.98,bottom=.22,top=.88,wspace=.28)
    fig.savefig(a.out,dpi=165,bbox_inches='tight',facecolor=fig.get_facecolor());plt.close(fig)

if __name__=='__main__':main()
