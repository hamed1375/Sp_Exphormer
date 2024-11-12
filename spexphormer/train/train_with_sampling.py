import logging
import time

import numpy as np
import torch
import torch.nn.functional as F
import random

from torch_geometric.graphgym.checkpoint import load_ckpt, save_ckpt, clean_ckpt
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.register import register_train
from torch_geometric.graphgym.utils.epoch import is_eval_epoch, is_ckpt_epoch
from time import perf_counter

from spexphormer.utils import cfg_to_dict, flatten_dict, make_wandb_name
from tqdm import tqdm

from spexphormer.train.neighbor_sampler import NeighborSampler



def train_epoch(logger, loader, model, optimizer, scheduler, batch_accumulation, neighbor_sampler):
    model.train()
    optimizer.zero_grad()
    time_start = time.time()

    train_nodes = torch.where(loader.dataset.train_mask)[0].tolist()
    random.shuffle(train_nodes)
    batch_size = cfg.train.batch_size

    n = len(train_nodes)
    num_iter = n//batch_size
    if n % batch_size:
        num_iter += 1
    for iter in tqdm(range(num_iter)):
        end_idx = min((iter+1) * batch_size, n)
        core_nodes = train_nodes[iter * batch_size:end_idx]

        batch = neighbor_sampler.make_batch(core_nodes=core_nodes)

        batch.split = 'train'
        batch.to(torch.device(cfg.accelerator))
        pred, true = model(batch)
        loss, pred_score = compute_loss(pred, true)
        _true = true.detach().to('cpu', non_blocking=True)
        _pred = pred_score.detach().to('cpu', non_blocking=True)
        loss.backward()

        # Parameters update after accumulating gradients for given num. batches.
        if ((iter + 1) % batch_accumulation == 0) or (iter + 1 == len(loader)):
            if cfg.optim.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        logger.update_stats(true=_true,
                            pred=_pred,
                            loss=loss.detach().cpu().item(),
                            lr=scheduler.get_last_lr()[0],
                            time_used=time.time() - time_start,
                            params=cfg.params,
                            dataset_name=cfg.dataset.name)
        time_start = time.time()


@torch.no_grad()
def eval_epoch(logger, loader, model, neighbor_sampler, split='val'):
    model.eval()
    time_start = time.time()
    
    data_mask = getattr(loader.dataset, f'{split}_mask')
    nodes = torch.where(data_mask)[0].tolist()
    random.shuffle(nodes)
    batch_size = cfg.train.batch_size

    n = len(nodes)
    num_iter = n // batch_size
    if n % batch_size:
        num_iter += 1
    process_times = []
    for iter in tqdm(range(num_iter)):
        end_idx = min((iter+1) * batch_size, n)
        core_nodes = nodes[iter * batch_size:end_idx]

        start_time = time.perf_counter()
        batch = neighbor_sampler.make_batch(core_nodes=core_nodes)
        end_time = time.perf_counter()
        process_times.append(end_time - start_time)

        batch.split = split
        batch.to(torch.device(cfg.accelerator))
        if cfg.gnn.head == 'inductive_edge':
            pred, true, extra_stats = model(batch)
        else:
            pred, true = model(batch)
            extra_stats = {}

        loss, pred_score = compute_loss(pred, true)
        _true = true.detach().to('cpu', non_blocking=True)
        _pred = pred_score.detach().to('cpu', non_blocking=True)
        logger.update_stats(true=_true,
                            pred=_pred,
                            loss=loss.detach().cpu().item(),
                            lr=0, time_used=time.time() - time_start,
                            params=cfg.params,
                            dataset_name=cfg.dataset.name,
                            **extra_stats)
        time_start = time.time()
    
    # print('average sampling time: ', np.mean(process_times))


@register_train('custom_with_sampling')
def custom_train_with_sampling(loggers, loaders, model, optimizer, scheduler):
    """
    Customized training pipeline.

    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler

    """

    scores, edge_index, edge_attr = read_scores()
    P, adj_eidx = separate_scores_by_node(scores, edge_index)
    neighbor_sampler = NeighborSampler(loaders[0].dataset, edge_index, edge_attr, P, adj_eidx, 
                                       deg=cfg.train.edge_sample_num_neighbors, num_layers=cfg.gt.layers,
                                       sampler=cfg.train.spexphormer_sampler)
    if cfg.train.spexphormer_sampler.startswith('graph'):
        resample_on_epoch = True
    else:
        resample_on_epoch = False

    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler,
                                cfg.train.epoch_resume)
    if start_epoch == cfg.optim.max_epoch:
        logging.info('Checkpoint found, Task already done')
    else:
        logging.info('Start from epoch %s', start_epoch)

    if cfg.wandb.use:
        try:
            import wandb
        except:
            raise ImportError('WandB is not installed.')
        if cfg.wandb.name == '':
            wandb_name = make_wandb_name(cfg)
        else:
            wandb_name = cfg.wandb.name
        run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project,
                         name=wandb_name)
        run.config.update(cfg_to_dict(cfg))

    num_splits = len(loggers)
    split_names = ['val', 'test']
    full_epoch_times = []
    perf = [[] for _ in range(num_splits)]

    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        cfg.train.cur_epoch = cur_epoch
        start_time = time.perf_counter()

        if resample_on_epoch and cur_epoch % cfg.train.resampling_epochs == 0:
            neighbor_sampler.all_nodes_neighbor_sampler()

        train_epoch(loggers[0], loaders[0], model, optimizer, scheduler,
                    cfg.optim.batch_accumulation, 
                    neighbor_sampler)
        perf[0].append(loggers[0].write_epoch(cur_epoch))
        if is_eval_epoch(cur_epoch):
            for i in range(1, num_splits):
                eval_epoch(loggers[i], loaders[i], model,
                           neighbor_sampler,
                           split=split_names[i - 1])
                perf[i].append(loggers[i].write_epoch(cur_epoch))
        else:
            for i in range(1, num_splits):
                perf[i].append(perf[i][-1])

        val_perf = perf[1]
        if cfg.optim.scheduler == 'reduce_on_plateau':
            scheduler.step(val_perf[-1]['loss'])
        else:
            scheduler.step()
        full_epoch_times.append(time.perf_counter() - start_time)
        # Checkpoint with regular frequency (if enabled).
        if cfg.train.enable_ckpt and not cfg.train.ckpt_best \
                and is_ckpt_epoch(cur_epoch):
            save_ckpt(model, optimizer, scheduler, cur_epoch)

        if cfg.wandb.use:
            run.log(flatten_dict(perf), step=cur_epoch)

        # Log current best stats on eval epoch.
        if is_eval_epoch(cur_epoch):
            best_epoch = np.array([vp['loss'] for vp in val_perf]).argmin()
            best_train = best_val = best_test = ""
            if cfg.metric_best != 'auto':
                # Select again based on val perf of `cfg.metric_best`.
                m = cfg.metric_best
                best_epoch = getattr(np.array([vp[m] for vp in val_perf]),
                                     cfg.metric_agg)()
                if m in perf[0][best_epoch]:
                    best_train = f"train_{m}: {perf[0][best_epoch][m]:.4f}"
                else:
                    # Note: For some datasets it is too expensive to compute
                    # the main metric on the training set.
                    best_train = f"train_{m}: {0:.4f}"
                best_val = f"val_{m}: {perf[1][best_epoch][m]:.4f}"
                best_test = f"test_{m}: {perf[2][best_epoch][m]:.4f}"

                if cfg.wandb.use:
                    bstats = {"best/epoch": best_epoch}
                    for i, s in enumerate(['train', 'val', 'test']):
                        bstats[f"best/{s}_loss"] = perf[i][best_epoch]['loss']
                        if m in perf[i][best_epoch]:
                            bstats[f"best/{s}_{m}"] = perf[i][best_epoch][m]
                            run.summary[f"best_{s}_perf"] = \
                                perf[i][best_epoch][m]
                        for x in ['hits@1', 'hits@3', 'hits@10', 'mrr']:
                            if x in perf[i][best_epoch]:
                                bstats[f"best/{s}_{x}"] = perf[i][best_epoch][x]
                    run.log(bstats, step=cur_epoch)
                    run.summary["full_epoch_time_avg"] = np.mean(full_epoch_times)
                    run.summary["full_epoch_time_sum"] = np.sum(full_epoch_times)
            # Checkpoint the best epoch params (if enabled).
            if cfg.train.enable_ckpt and cfg.train.ckpt_best and \
                    best_epoch == cur_epoch:
                save_ckpt(model, optimizer, scheduler, cur_epoch)
                if cfg.train.ckpt_clean:  # Delete old ckpt each time.
                    clean_ckpt()
            logging.info(
                f"> Epoch {cur_epoch}: took {full_epoch_times[-1]:.1f}s "
                f"(avg {np.mean(full_epoch_times):.1f}s) | "
                f"Best so far: epoch {best_epoch}\t"
                f"train_loss: {perf[0][best_epoch]['loss']:.4f} {best_train}\t"
                f"val_loss: {perf[1][best_epoch]['loss']:.4f} {best_val}\t"
                f"test_loss: {perf[2][best_epoch]['loss']:.4f} {best_test}"
            )
            if hasattr(model, 'trf_layers'):
                # Log SAN's gamma parameter values if they are trainable.
                for li, gtl in enumerate(model.trf_layers):
                    if torch.is_tensor(gtl.attention.gamma) and \
                            gtl.attention.gamma.requires_grad:
                        logging.info(f"    {gtl.__class__.__name__} {li}: "
                                     f"gamma={gtl.attention.gamma.item()}")
    logging.info(f"Avg time per epoch: {np.mean(full_epoch_times):.2f}s")
    logging.info(f"Total train loop time: {np.sum(full_epoch_times) / 3600:.2f}h")

    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()
    # close wandb
    if cfg.wandb.use:
        run.finish()
        run = None

    logging.info('Task done, results saved in %s', cfg.run_dir)


def read_scores(ref_dim=None, seed=0):
    if ref_dim is None:
        if cfg.dataset.name.startswith('ogbn') or cfg.dataset.name == 'pokec':
            ref_dim = 8
        else:
            ref_dim = 4
    edge_index = np.load(f'Attention_scores/{cfg.dataset.name}/edges.npy')
    edge_attr = np.load(f'Attention_scores/{cfg.dataset.name}/edge_attr.npy')
    scores = [] 
    for i in range(cfg.gt.layers):
        scores.append(np.load(f'Attention_scores/{cfg.dataset.name}/seed{seed}_h{ref_dim}_layer_{i}.npy').squeeze())
    return scores, edge_index, edge_attr


def separate_scores_by_node(scores, edges):
    num_nodes = np.max(edges) + 1
    neighbors_edge_idx = [[] for _ in range(num_nodes)]
    P = [[] for _ in range(num_nodes)]
    
    for l in range(cfg.gt.layers):
        for i in range(edges.shape[1]):
            v = edges[1][i]
            if l==0:
                neighbors_edge_idx[v].append(i)
            P[v].append(scores[l][i])

    for v in range(num_nodes):
        P[v] = np.array(P[v]).reshape((cfg.gt.layers, -1))
        P[v] = P[v]/(P[v].sum(axis=1)[:, np.newaxis])
        neighbors_edge_idx[v] = np.array(neighbors_edge_idx[v])

    return P, neighbors_edge_idx