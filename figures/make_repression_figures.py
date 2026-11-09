"""Regenerate the two Repressor figures from archived numerical outputs.

No fitting or new synthetic data. Run with --data-dir containing
response_plot_data.npz and repression_grid_independent_validation.json.
"""
from pathlib import Path
import argparse, json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.labelsize':7,
    'axes.titlesize':7.5,'xtick.labelsize':6.4,'ytick.labelsize':6.4,
    'legend.fontsize':6.1,'axes.linewidth':.55,'lines.linewidth':1.1,
    'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})
RED='#B33030';GREY='#697078'
DOSE_COLORS={20:'#B33030',50:'#236C8C',100:'#2F7040'}
TITLES={'model_1':'Model I','model_2':'Model II'}

def clean(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(width=.5,length=2.5,pad=2)
    ax.grid(axis='y',color='#E4E6E8',lw=.4,zorder=0)
    ax.set_axisbelow(True)

def save(fig,out,name):
    fig.savefig(out/(name+'.pdf'))
    fig.savefig(out/(name+'.png'),dpi=240)
    plt.close(fig)

def loading(data,out):
    rows=json.loads((data/'repression_grid_independent_validation.json').read_text())['rows']
    fig,axs=plt.subplots(1,2,figsize=(3.45,2.12),sharey=True)
    fig.subplots_adjust(left=.13,right=.98,bottom=.23,top=.70,wspace=.20)
    for i,mech in enumerate(['model_1','model_2']):
        ax=axs[i]
        for dose,color in DOSE_COLORS.items():
            rr=sorted((r for r in rows if r['mechanism']==mech and r['dose_ng']==dose),key=lambda r:r['loading'])
            assert len(rr)==9
            xx=[r['loading'] for r in rr]
            ax.plot(xx,[100*r['Dinf'] for r in rr],color=color,marker='o',ms=2.5,lw=.95)
            ax.plot(xx,[100*abs(r['E']) for r in rr],color=color,marker='s',ms=2.2,lw=.9,ls=(0,(3,2)))
        ax.axhline(10,color='#686D73',ls=(0,(1,2)),lw=.7)
        ax.set(xlim=(0,1),ylim=(0,25),xticks=[0,.5,1],yticks=[0,10,20],xlabel=r'Target loading $\epsilon$')
        ax.set_title(TITLES[mech],pad=4)
        ax.text(-.04,1.04,f'({chr(97+i)})',transform=ax.transAxes,fontweight='bold',ha='right',va='bottom')
        clean(ax)
    axs[0].set_ylabel('Relative discrepancy (%)')
    doses=[Line2D([],[],color=c,lw=1.2,label=f'{q} ng') for q,c in DOSE_COLORS.items()]
    metrics=[Line2D([],[],color='#333333',marker='o',ms=2.5,label=r'Waveform $D^\infty_{48}$'),
             Line2D([],[],color='#333333',ls=(0,(3,2)),marker='s',ms=2.5,label=r'Area $|E_{48}|$')]
    fig.legend(handles=doses,loc='upper center',bbox_to_anchor=(.55,1.005),frameon=False,ncol=3,columnspacing=1.3)
    fig.legend(handles=metrics,loc='upper center',bbox_to_anchor=(.55,.895),frameon=False,ncol=2,columnspacing=1.)
    save(fig,out,'repression_loading_effects')

def response(data,out):
    saved=np.load(data/'response_plot_data.npz')
    peaks=json.loads((data/'response_peak_normalization.json').read_text())
    fig,axs=plt.subplots(2,2,figsize=(3.45,2.95),sharex=True)
    fig.subplots_adjust(left=.15,right=.98,bottom=.12,top=.85,wspace=.28,hspace=.25)
    for col,mech in enumerate(['model_1','model_2']):
        t=saved[mech+'_time'];ref=saved[mech+'_omitted'];loaded=saved[mech+'_included']
        peak=float(peaks[mech])
        for yy,color,style in [(ref[3]/peak,GREY,(0,(4,2))),(loaded[3]/peak,RED,'-')]:
            axs[0,col].plot(t,yy,color=color,ls=style,zorder=3)
        axs[0,col].set_ylim(0,1.08);axs[0,col].set_yticks([0,.5,1.])
        axs[0,col].set_title(TITLES[mech],pad=4)
        smin=min(ref[9].min(),loaded[9].min())*1.08
        axs[1,col].fill_between(t,smin,0,where=loaded[8]>0,color='#E8E9EB',interpolate=True,zorder=0)
        axs[1,col].plot(t,ref[9],color=GREY,ls=(0,(4,2)),zorder=3)
        axs[1,col].plot(t,loaded[9],color=RED,zorder=4)
        axs[1,col].set_ylim(smin,3);axs[1,col].axhline(0,color='#333333',lw=.45)
        pos=t[loaded[8]>0]
        if len(pos):axs[1,col].text((pos.min()+pos.max())/2,smin*.13,r'$\partial_\theta Y>0$',fontsize=6,ha='center',va='center',color='#55585E')
        axs[1,col].set_xlabel('Time (h)')
        for row in range(2):
            ax=axs[row,col];clean(ax);ax.set_xlim(0,48);ax.set_xticks([0,24,48])
            ax.text(-.04,1.03,f'({chr(97+row*2+col)})',transform=ax.transAxes,fontsize=7,fontweight='bold',ha='right',va='bottom')
    axs[0,0].set_ylabel('Output / reference peak');axs[1,0].set_ylabel(r'Cumulative slope $S_T$')
    handles=[Line2D([],[],color=GREY,ls=(0,(4,2)),label='Demand omitted'),Line2D([],[],color=RED,label='Demand included')]
    fig.legend(handles=handles,loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.55,.99),columnspacing=1.2,handlelength=2.)
    save(fig,out,'repression_response_metrics')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).parent)
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    loading(args.data_dir,args.output_dir);response(args.data_dir,args.output_dir)
    paths=[args.data_dir/'repression_grid_independent_validation.json',args.data_dir/'response_plot_data.npz',args.data_dir/'response_peak_normalization.json']
    (args.output_dir/'figure_data_provenance.json').write_text(json.dumps({'scope':'Ordinary fixed-fit numerical predictions; connecting curves do not certify intermediate doses or loadings.',
        'inputs':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'typography':'Generic model labels are used consistently in the external plots.'},indent=2)+'\n')

if __name__=='__main__':main()
