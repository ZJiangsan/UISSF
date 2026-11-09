#!/usr/bin/env python3
"""Generate deterministic, fully independent Slurm jobs for UISSF manuscript studies.

This script generates job files and a manifest; it does not submit by default.
Use --submit to schedule only one dataset+phase batch deliberately.
"""
import argparse,csv,json,shlex,subprocess
from pathlib import Path

DATA={
'chikusei':{'hsi':'data/chikusei_1000x1000x31.npy','suffix':'chikusei'},
'cave':{'hsi':'data/cave_balloons_512x512x31.npy','suffix':'cave'},
'icvl':{'hsi':'data/icvl_colorcheck_800x800x31.npy','suffix':'icvl'},
}
ENC_ROOT='/trinity/home/jizh/UISSF/checkpoints/decomposition'
SRF='assets/uissf_srf_np_31x3.npy'
BASE=Path('/trinity/home/jizh/UISSF_release')
RUN_BASE=Path('/trinity/home/jizh/UISSF_runs/manuscript_reproduction')

def experiments(phase,seeds):
    if phase=='main':return [dict(tag='combined_workingone',loss='all',variant='workingone',residual=True,seed=0,snr=None,permute=False)]
    if phase=='loss':return [dict(tag=f'loss_{k}',loss=k,variant='workingone',residual=True,seed=0,snr=None,permute=False) for k in ('l21','ergas','sam','l21_ergas','l21_sam','ergas_sam','all')]
    if phase=='loss_manuscript':return [dict(tag=f'manuscript_{k}',loss=k,variant='workingone',residual=True,seed=0,snr=None,permute=False) for k in ('l21','ergas','sam','l21_ergas','l21_sam','ergas_sam','all')]
    if phase=='residual':return [dict(tag=f'residual_{int(r)}',loss='all',variant='ablation',residual=r,seed=0,snr=None,permute=False) for r in (True,False)]
    if phase=='permutation':return [dict(tag=f'permuted_{int(p)}',loss='all',variant='ablation',residual=True,seed=0,snr=None,permute=p) for p in (False,True)]
    if phase=='noise':return [dict(tag=f'noise_{s if s else "clean"}_seed{seed}',loss='all',variant='ablation',residual=True,seed=seed,snr=s,permute=False) for s in (None,40,30,20) for seed in range(seeds)]
    raise ValueError(phase)

def generate(args):
    project=Path(args.project);outroot=Path(args.output_root)
    ds=DATA[args.dataset];hsi=project/ds['hsi'];crf=project/SRF
    encoder=Path(ENC_ROOT)/f'unsupervised_CRF_learning_encoder_uav_chikusei_CE_40_endmember_sam_3spc31_g54_02_{ds["suffix"]}.pth'
    decoder=Path(ENC_ROOT)/f'unsupervised_CRF_learning_decoder_uav_chikusei_CE_40_endmember_sam_3spc31_g54_02_{ds["suffix"]}.pth'
    if args.phase=='loss' and args.objective!='original':
        raise ValueError('Fixed original composite requires --objective original')
    if getattr(args,"pure_gram_loss",False) and args.phase!='loss':
        raise ValueError('--pure-gram-loss only supports --phase loss')
    if args.check_inputs:
        missing=[str(p) for p in (project/'scripts/uissf_cli.py',project/'scripts/evaluate_checkpoint_comparison.py',hsi,crf,encoder,decoder) if not p.is_file()]
        if missing:raise FileNotFoundError('Missing input files: '+'; '.join(missing))
    folder=outroot/args.dataset/args.phase;folder.mkdir(parents=True,exist_ok=True)
    jobs=folder/'jobs';jobs.mkdir(parents=True,exist_ok=True)
    rows=[]
    for exp in experiments(args.phase,args.seeds):
        target=folder/exp['tag'];log=folder/(exp['tag']+'_%j.log')
        cmd=[str(project/'scripts/uissf_cli.py'),'map','--hsi',str(hsi),'--crf',str(crf),'--encoder',str(encoder),'--decoder',str(decoder),'--output',str(target),'--epochs',str(args.epochs),'--lr','1e-4','--device','cuda','--mode','legacy','--loss-config',exp['loss'],'--loss-variant',exp['variant'],'--objective',args.objective,'--eval-every',str(args.eval_every),'--save-every',str(args.save_every),'--log-every',str(args.log_every),'--reset-adam-each-epoch','--seed',str(exp['seed'])]
        if getattr(args,"pure_gram_loss",False):
            cmd.extend(['--reconstruction-weight','0','--entropy-weight','0'])
        if not exp['residual']:cmd.append('--no-residual')
        if exp['permute']:cmd.append('--permute')
        if exp['snr'] is not None:cmd+=['--snr',str(exp['snr'])]
        evalcmd=['scripts/evaluate_checkpoint_comparison.py','--hsi',str(hsi),'--crf',str(crf),'--encoder',str(encoder),'--decoder',str(decoder),'--checkpoint',str(target/'training_state.pth'),'--output',str(target/'evaluation'),'--device','cuda','--mode','legacy']
        lines=['#!/bin/bash',f'#SBATCH --job-name=UISSF_{args.dataset[:3]}_{exp["tag"][:20]}','#SBATCH --partition=compute-group','#SBATCH --gres=gpu:l40s:1','#SBATCH --cpus-per-task=8','#SBATCH --mem=32G',f'#SBATCH --time={args.walltime}',f'#SBATCH --output={log}','set -euo pipefail',f'cd {shlex.quote(str(project))}',f'mkdir -p {shlex.quote(str(target))}',f'PY={shlex.quote(str(args.python))}',f'if test -f {shlex.quote(str(target/"training_state.pth"))}; then RESUME="--resume"; else RESUME=""; fi',f'$PY {shlex.join(cmd)} $RESUME',f'PYTHONPATH=. $PY {shlex.join(evalcmd)}','echo "COMPLETE: '+str(target)+'"']
        lines.insert(lines.index(f'$PY {shlex.join(cmd)} $RESUME'),'export PYTHONPATH=.')
        script=jobs/(exp['tag']+'.sbatch');script.write_text('\n'.join(lines)+'\n');script.chmod(0o755)
        rows.append({'dataset':args.dataset,'phase':args.phase,'experiment':exp['tag'],'objective':args.objective,'loss_config':exp['loss'],'loss_variant':exp['variant'],'seed':exp['seed'],'snr':exp['snr'],'residual':exp['residual'],'permuted':exp['permute'],'epochs':args.epochs,'script':str(script),'output':str(target),'hsi':str(hsi),'encoder':str(encoder),'decoder':str(decoder)})
    (folder/'manifest.json').write_text(json.dumps({'dataset':args.dataset,'phase':args.phase,'experiments':rows,'pure_gram_loss':getattr(args,"pure_gram_loss",False),'auxiliary_weights':{'reconstruction':0.0 if getattr(args,"pure_gram_loss",False) else None,'entropy':0.0 if getattr(args,"pure_gram_loss",False) else None},'exclusions':['External WLRTR/u2MDN and EDIP-Net baselines not included'], 'warning':'Loss phase uses unchanged workingone full composite; only loss component switches are zeroed. Requires original datasets/checkpoints on Trinity; training is opt-in'},indent=2))
    with (folder/'manifest.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    print(f'Generated {len(rows)} scripts in {jobs}')
    if args.submit:
        for row in rows:
            completed=subprocess.run(['sbatch',row['script']],check=True,text=True,capture_output=True)
            print(row['experiment'],completed.stdout.strip())
    else:
        print('Not submitted. Use --submit to schedule this phase or sbatch a particular job.')
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=DATA.keys(),required=True);p.add_argument('--phase',choices=['main','loss','loss_manuscript','residual','permutation','noise'],required=True);p.add_argument('--epochs',type=int,default=100000);p.add_argument('--seeds',type=int,default=3);p.add_argument('--eval-every',type=int,default=10000);p.add_argument('--save-every',type=int,default=1000);p.add_argument('--log-every',type=int,default=1000);p.add_argument('--objective',choices=['original','manuscript'],default='original');p.add_argument('--walltime',default='24:00:00');p.add_argument('--project',default=str(BASE));p.add_argument('--output-root',default=str(RUN_BASE));p.add_argument('--python',default='/trinity/home/jizh/venvs/nerfstudio/bin/python');p.add_argument('--check-inputs',action='store_true');p.add_argument('--submit',action='store_true');p.add_argument('--pure-gram-loss',action='store_true',help='For loss ablation: set auxiliary reconstruction and entropy weights to zero for all seven runs; preserve fixed Gram/diagonal coefficients');args=p.parse_args();
    if args.epochs<1 or args.seeds<1:raise ValueError('epochs and seeds must be positive')
    if args.objective=='manuscript' and args.phase!='loss_manuscript':raise ValueError('Use manuscript objective only in loss_manuscript phase')
    generate(args)
if __name__=='__main__':main()
