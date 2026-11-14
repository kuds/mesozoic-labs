"""Render top projections of the A10/C10 continuous extraction sweep."""
from itertools import product
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull
from screen import Installed, BOX, MESH, CYLINDER, CAPSULE, CORNERS


HERE=Path(__file__).resolve().parent
def hull(points):
    pts=np.asarray(points)[:,:2]*1000
    if len(pts)>2 and np.linalg.matrix_rank(pts-pts[0])==2:
        return pts[ConvexHull(pts).vertices]
    return pts


def poly(ax,pts,**style):
    ax.add_patch(Polygon(hull(pts),closed=True,**style))


def geom_outline(q,name):
    if name in q.mesh:
        return q.mesh[name].reshape(-1,3)
    i=q.names[name];p,r,s=q.pose(name);kind=q.m.geom_type[i]
    if kind==BOX:
        v=CORNERS*q.true_half(name)
    elif kind in (CYLINDER,CAPSULE):
        ring=np.linspace(0,2*np.pi,48,endpoint=False)
        if kind==CYLINDER:
            v=np.array([[s[0]*np.cos(t),s[0]*np.sin(t),sg*s[1]] for t,sg in product(ring,[-1,1])])
        else:
            v=np.array([[s[0]*np.cos(t)*np.cos(f),s[0]*np.sin(t)*np.cos(f),sg*s[1]+s[0]*np.sin(f)] for t,f,sg in product(ring,np.linspace(-np.pi/2,np.pi/2,9),[-1,1])])
    else:
        v=CORNERS*s
    return v@r.T+p


def main():
    data=json.loads((HERE/'screen.json').read_text())
    fig,axs=plt.subplots(1,2,figsize=(11.6,7),dpi=170)
    fig.patch.set_facecolor('#fafbf8')
    for ax,model,title in zip(axs,['A10','C10'],['Tapered flank · longitudinal pack','Open rib frame · transverse pack']):
        q=Installed(HERE.parent/'model'/'generated'/f'{model}.xml')
        p,r,s=q.pose('battery_pack');half=q.true_half('battery_pack')+.008
        left=q.d.xmat[q.m.body('pelvis').id].reshape(3,3)[:,1]
        verts=CORNERS*half@r.T+p
        path=np.vstack([verts,verts+left*.24])
        b=data[model]['battery_routes']['left'];blockers=b['blockers']
        for n in q.names:
            if n and n.startswith('cover_') and n in q.mesh:
                if np.max(q.mesh[n][:,:,0])<-.31 or np.min(q.mesh[n][:,:,0])>.19:continue
                poly(ax,geom_outline(q,n),facecolor='none',edgecolor='#bec4bb',alpha=.6,linewidth=.55,zorder=1)
        poly(ax,path,facecolor='#66a5ac',edgecolor='#256a73',alpha=.23,linewidth=1.4,zorder=2)
        for n in blockers:
            if n in q.mesh and n.startswith('cover_'):
                poly(ax,geom_outline(q,n),facecolor='none',edgecolor='#3c78a3',linewidth=1.15,linestyle=':',zorder=3)
            else:
                poly(ax,geom_outline(q,n),facecolor='#d07959',edgecolor='#ad4f32',alpha=.56,linewidth=.7,zorder=3)
        poly(ax,geom_outline(q,'battery_pack'),facecolor='#243d43',edgecolor='#0f2c35',alpha=.95,linewidth=1.1,zorder=4)
        poly(ax,verts+left*.24,facecolor='none',edgecolor='#256a73',linestyle='--',linewidth=1.1,zorder=3)
        x,y=p[:2]*1000
        ax.annotate('',xy=(x,y+228),xytext=(x,y+55),arrowprops=dict(arrowstyle='-|>',color='#256a73',lw=1.6),zorder=5)
        ax.text(x+8,y+148,'240 mm',color='#256a73',fontsize=9,rotation=90,ha='left',va='center')
        count=len(b['remaining_after_removing_all_intersecting_covers'])
        ax.set_title(f'{model}   {title}',loc='left',fontsize=11.2,fontweight='bold',pad=9,color='#173c45')
        ax.text(.02,.965, 'Clear after panel removal' if not count else f'{count} fixed items cross guarded stroke', 
                transform=ax.transAxes,ha='left',va='top',fontsize=9.4,fontweight='bold',color='#256a73' if not count else '#a34a31',
                bbox=dict(facecolor='#fafbf8',edgecolor='none',pad=2.5))
        ax.set_aspect('equal');ax.set_xlim(-310,175);ax.set_ylim(-140,355)
        ax.set_xticks([-300,-200,-100,0,100]);ax.set_yticks([-100,0,100,200,300])
        ax.tick_params(labelsize=8,colors='#567074',width=.6)
        ax.set_xlabel('Fore–aft, mm  ·  head to the right',fontsize=9)
        ax.set_facecolor('#fafbf8');ax.grid(alpha=.13,color='#5d7875');ax.set_axisbelow(True)
        for spine in ax.spines.values():spine.set_color('#a2b1aa');spine.set_linewidth(.7)
    axs[0].set_ylabel('Robot-left displacement, mm',fontsize=9)
    fig.subplots_adjust(left=.075,right=.982,top=.826,bottom=.207,wspace=.13)
    fig.text(.076,.95,'Battery service in plan view',weight='bold',fontsize=22,color='#163943',va='top')
    fig.text(.077,.883,'Guarded straight pull from the installed cassette. Each panel uses the same physical scale.',fontsize=10.5,color='#485f61')
    handles=[Line2D([],[],lw=6,color='#243d43',label='Installed pack'),Line2D([],[],lw=7,color='#9fc4c6',label='Guarded sweep'),
            Line2D([],[],lw=6,color='#c67d60',label='Fixed obstruction / allocation'),Line2D([],[],ls=':',lw=2,color='#3c78a3',label='Removable cover')]
    fig.legend(handles=handles,ncol=4,loc='lower left',bbox_to_anchor=(.069,.102),frameon=False,fontsize=9,handlelength=1.5)
    fig.text(.076,.05,'Projected part bounds are shown for orientation. Intersections are computed in 3D over the full stroke at the review pose.',fontsize=9,color='#485f61')
    fig.savefig(HERE/'service_paths.png',bbox_inches='tight',pad_inches=.18,facecolor=fig.get_facecolor());plt.close(fig)

if __name__=='__main__':
    main()
