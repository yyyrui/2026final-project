"""Unmodified TACTiS2 core, one set of embeddings and weights for every series."""
from argparse import Namespace
from copy import deepcopy
import json
import random
import sys
import time

import numpy as np
import pandas as pd
import torch

from comparison.full import atomic_json, result_frame, scores, score_sums, seed_all
from comparison.models import ROOT, load_chronos, predict_chronos


def save_checkpoint(path, checkpoint):
    temporary=path.with_suffix('.pt.tmp')
    torch.save(checkpoint,temporary)
    temporary.replace(path)


def build_model(series, config, device, stage=1):
    sys.path.insert(0,str(ROOT/'TACTiS2/TACTiS'))
    from tactis.model.tactis import TACTiS
    model=TACTiS(num_series=series,bagging_size=min(series,config['training']['bagging_size']),
                 **deepcopy(config['model']))
    if stage==2: model.initialize_stage2()
    return model.to(device)


def set_stage_parameters(model, stage):
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(stage==1 or 'copula' in name)


def windows(values, ends, context, horizon, device):
    hist=torch.stack([values[:,end-context:end] for end in ends]).to(device)
    future=torch.stack([values[:,end:end+horizon] for end in ends]).to(device)
    batch=len(ends)
    ht=torch.arange(context,device=device)[None,None,:].expand(batch,-1,-1)
    pt=torch.arange(context,context+horizon,device=device)[None,None,:].expand(batch,-1,-1)
    return ht,hist,pt,future


def validation(model, values, ends, config, device):
    # All series participate in the SAME joint density for each validation window.
    previous=model.bagging_size
    model.bagging_size=None
    model.eval()
    losses=[]
    try:
        with torch.no_grad():
            for end in ends:
                marginal,copula=model.loss(*windows(values,[end],config['context_length'],config['horizon'],device))
                loss=(copula-marginal).mean()
                if not torch.isfinite(loss): raise ValueError('Nonfinite unbagged validation loss')
                losses.append(loss.item())
    finally: model.bagging_size=previous
    return float(np.mean(losses))


def rng_state(rng):
    return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                numpy=rng.bit_generator.state,python=random.getstate())


def restore_rng(state,rng):
    torch.set_rng_state(state['torch'].cpu())
    if torch.cuda.is_available() and state['cuda']: torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])
    rng.bit_generator.state=state['numpy']
    random.setstate(state['python'])


def train(panel,config,args,output,identity):
    parameters=config['training']
    horizon,context=config['horizon'],config['context_length']
    cutoff=panel.train.shape[1]-config['validation_windows']*horizon
    if cutoff<context+horizon: raise ValueError('Insufficient days before validation cutoff')
    val_ends=list(range(cutoff,panel.train.shape[1]-horizon+1,horizon))
    values=torch.from_numpy(panel.train)
    seed_all(args.seed)
    rng=np.random.default_rng(args.seed)
    path=output/'last.pt'
    state=torch.load(path,map_location=args.device,weights_only=False) if args.resume and path.exists() else None
    if state and state['identity']!=identity: raise ValueError('Checkpoint identity mismatch')
    stage=state['stage'] if state else 1
    model=build_model(len(panel.keys),config,args.device,stage)
    if state: model.load_state_dict(state['model'])
    logs=state['logs'] if state else []
    for current in range(stage,3):
        if current!=stage: model.initialize_stage2(); model.to(args.device)
        set_stage_parameters(model,current)
        cfg=parameters[f'stage{current}']
        optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                   lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        initial=0; best=float('inf'); patience=0; seconds=0.; stage_done=False
        if state and current==stage:
            optimizer.load_state_dict(state['optimizer'])
            initial=state['epoch']+1; best=state['best']; patience=state['patience']; seconds=state['training_seconds']
            stage_done=state['stage_done']; restore_rng(state['rng'],rng)
        for epoch in range(initial,parameters['max_epochs_per_stage']):
            if stage_done: break
            model.train(); total=0.; count=0
            started=time.perf_counter()
            for _ in range(parameters['steps_per_epoch']):
                ends=rng.integers(context,cutoff-horizon+1,size=parameters['batch_size'])
                optimizer.zero_grad(set_to_none=True)
                marginal,copula=model.loss(*windows(values,ends,context,horizon,args.device))
                loss=(-marginal if current==1 else copula).mean()
                if not torch.isfinite(loss): raise ValueError('Nonfinite joint training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip_gradient'],error_if_nonfinite=True)
                optimizer.step(); total+=loss.item(); count+=1
                if seconds+time.perf_counter()-started>=parameters['max_training_hours_per_stage']*3600: break
            if args.device=='cuda': torch.cuda.synchronize()
            seconds+=time.perf_counter()-started
            val=validation(model,values,val_ends,config,args.device)
            if val<best:
                best=val; patience=0
                save_checkpoint(output/f'best_stage{current}.pt',dict(model=model.state_dict(),epoch=epoch,validation_loss=val,stage=current))
            else: patience+=1
            stage_done=(patience>=parameters['patience'] or seconds>=parameters['max_training_hours_per_stage']*3600
                        or epoch+1>=parameters['max_epochs_per_stage'])
            logs.append(dict(stage=current,epoch=epoch+1,train_loss=total/count,validation_loss=val,
                             optimizer_steps=count,training_seconds=seconds))
            print(logs[-1],flush=True)
            atomic_json(output/'training_log.json',logs)
            save_checkpoint(path,dict(identity=identity,stage=current,model=model.state_dict(),optimizer=optimizer.state_dict(),
                                      epoch=epoch,best=best,patience=patience,training_seconds=seconds,stage_done=stage_done,
                                      logs=logs,rng=rng_state(rng)))
            atomic_json(output/'status.json',dict(status='training',stage=current,epoch=epoch+1,validation_loss=val))
        model.load_state_dict(torch.load(output/f'best_stage{current}.pt',map_location=args.device,weights_only=False)['model'])
        state=None
    save_checkpoint(output/'tactis.pt',dict(identity=identity,stage=2,config=config,model=model.state_dict()))
    return model


def run(panel,config,args,output,identity):
    if (output/'SUCCESS').exists():
        print('Already complete; refusing to repeat a completed experiment.',flush=True)
        return
    if args.resume and (output/'tactis.pt').exists():
        state=torch.load(output/'tactis.pt',map_location=args.device,weights_only=False)
        if state['identity']!=identity: raise ValueError('Final checkpoint mismatch')
        model=build_model(len(panel.keys),config,args.device,stage=2)
        model.load_state_dict(state['model'])
    else: model=train(panel,config,args,output,identity)
    model.eval(); model.bagging_size=None
    horizon,context=config['horizon'],config['context_length']
    samples_config=config['sampling']
    samples_dir=output/'joint_samples'; samples_dir.mkdir(exist_ok=True)
    torch_values=torch.from_numpy(panel.train[None,:,-context:]).to(args.device)
    ht=torch.arange(context,device=args.device)[None,None,:]
    pt=torch.arange(context,context+horizon,device=args.device)[None,None,:]
    samples=[]
    # Batching independent draws is exact; each call still sees all 50,000 series.
    for offset in range(0,samples_config['num_samples'],samples_config['samples_per_call']):
        file=samples_dir/f'{offset:05d}.npy'
        count=min(samples_config['samples_per_call'],samples_config['num_samples']-offset)
        if not file.exists():
            seed_all(args.seed+100000+offset)
            with torch.no_grad():
                result=model.sample(count,ht,torch_values,pt)[0,:,-horizon:,:].cpu().numpy()
            if not np.isfinite(result).all(): raise ValueError('Nonfinite TACTiS samples')
            with file.with_suffix('.tmp').open('wb') as stream: np.save(stream,result)
            file.with_suffix('.tmp').replace(file)
        result=np.load(file)
        if result.shape!=(len(panel.keys),horizon,count) or not np.isfinite(result).all(): raise ValueError('Invalid sample partition')
        samples.append(result)
    forecast=np.concatenate(samples,axis=-1)
    first=result_frame(panel,0,len(panel.keys),forecast)
    first.to_csv(output/'tactis2_predictions.csv',index=False)
    rows=[dict(model='tactis2_joint',**scores(score_sums(first.actual,first.prediction)))]
    del samples,forecast,model,torch_values
    if args.device=='cuda': torch.cuda.empty_cache()
    options=Namespace(**vars(args));options.context_length=context;options.horizon=horizon;options.num_samples=samples_config['num_samples']
    pipeline=load_chronos(options)
    folder=output/'chronos_batches';folder.mkdir(exist_ok=True)
    total=dict(count=0,absolute=0.,squared=0.,denominator=0.)
    with (output/'chronos_predictions.csv.tmp').open('w') as stream:
        for start in range(0,len(panel.keys),args.chronos_batch_size):
            end=min(start+args.chronos_batch_size,len(panel.keys)); file=folder/f'{start:05d}.parquet'
            if not file.exists():
                seed_all(args.seed+200000+start)
                forecast=predict_chronos(panel.train[start:end],options,pipeline=pipeline)
                frame=result_frame(panel,start,end,forecast)
                frame.to_parquet(file.with_suffix('.tmp'),index=False);file.with_suffix('.tmp').replace(file)
            from comparison.full import read_result
            frame,subtotal=read_result(file,panel,start,end)
            frame.to_csv(stream,index=False,header=start==0)
            for key,value in subtotal.items():total[key]+=value
    (output/'chronos_predictions.csv.tmp').replace(output/'chronos_predictions.csv')
    rows.append(dict(model='chronos',**scores(total)))
    pd.DataFrame(rows).to_csv(output/'metrics.csv',index=False)
    atomic_json(output/'status.json',dict(status='complete',series=len(panel.keys),scope=identity['scope']))
    (output/'SUCCESS').write_text(identity['scope']+'\n')
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
