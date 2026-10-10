"""Plot recorded callbacks; task-specific diagnostics, not benchmarks."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('<EVIDENCE>'))
    args=p.parse_args()
    trials=['flat_rewardv2_30m_s0','rough_rewardv2_10m_capacity320','rough_wider_reset20m_s0_v2']
    fig,axes=plt.subplots(len(trials),2,figsize=(12,9))
    for row,trial in enumerate(trials):
        path=args.root/trial/'result/progress.jsonl'
        if not path.exists():continue
        records=[json.loads(x) for x in path.read_text().splitlines()]
        x=[r['steps']/1e6 for r in records]
        for col,key in enumerate(['eval/episode_reward','eval/avg_episode_length']):
            axes[row,col].plot(x,[r['metrics'][key] for r in records],marker='o')
            axes[row,col].set(title=trial+'\n'+key,xlabel='Branch sampling steps (millions)')
            axes[row,col].grid(alpha=.25)
    fig.suptitle('Recorded callbacks: different task/reset distributions\nOnly 3 callbacks per branch; reward is not independent tracking success')
    fig.tight_layout(rect=(0,0,1,.94))
    fig.savefig(args.root/'training.png',dpi=150)
    plt.close(fig)


if __name__=='__main__':main()
