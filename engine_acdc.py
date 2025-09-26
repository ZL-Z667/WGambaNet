import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast
import time

from utils import test_single_volume_acdc

def train_one_epoch(train_loader,
                    model,
                    criterion,
                    optimizer,
                    scheduler,
                    epoch,
                    logger,
                    config,
                    scaler=None):
    """
    Train model for one epoch on ACDC dataset
    """
    stime = time.time()
    model.train()
    loss_list = []
    for iter, sample in enumerate(train_loader):
        optimizer.zero_grad()
        images, targets = sample['image'], sample['label']
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).long()
        if config.amp:
            with autocast():
                out = model(images)
                loss = criterion(out, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(images)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()
        loss_list.append(loss.item())
        now_lr = optimizer.state_dict()['param_groups'][0]['lr']
        mean_loss = np.mean(loss_list)
        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {loss.item():.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
    scheduler.step()
    etime = time.time()
    log_info = f'Finish one epoch train: epoch {epoch}, loss: {mean_loss:.4f}, time(s): {etime-stime:.2f}'
    print(log_info)
    logger.info(log_info)
    return mean_loss

def val_one_epoch(test_loader, model, epoch, logger, config, test_save_path=None):
    """
    Validate on ACDC dataset.
    Returns average DSC, RV, MYO, LV, HD95.
    """
    stime = time.time()
    model.eval()
    metrics_list = []
    with torch.no_grad():
        for sample in tqdm(test_loader):
            img, msk = sample['image'], sample['label']
            case_name = sample['patient_id'][0] if 'patient_id' in sample else f'case_{len(metrics_list)}'
            metrics, pred = test_single_volume_acdc(
                img, msk, model, classes=config.num_classes,
                patch_size=[config.input_size_h, config.input_size_w],
                test_save_path=test_save_path, case=case_name, z_spacing=config.z_spacing
            )
            metrics_list.append(list(metrics.values()))
            logger.info(
                f"Case {case_name}: DSC={metrics['DSC']:.4f} RV={metrics['RV']:.4f} MYO={metrics['MYO']:.4f} LV={metrics['LV']:.4f} HD95={metrics['HD95']:.4f}"
            )
    metrics_arr = np.array(metrics_list)
    avg_metrics = metrics_arr.mean(axis=0)
    etime = time.time()
    log_info = (
        f'val epoch: {epoch}, Mean DSC: {avg_metrics[0]:.4f} (RV:{avg_metrics[1]:.4f}, MYO:{avg_metrics[2]:.4f}, LV:{avg_metrics[3]:.4f}), Mean HD95: {avg_metrics[4]:.4f}, time(s): {etime-stime:.2f}'
    )
    print(log_info)
    logger.info(log_info)
    return avg_metrics

def test_pth(test_loader, model, logger, config, test_save_path, save_slices_dir=None):
    """
    Test final model on ACDC dataset and optionally save slice images.
    Returns average DSC, RV, MYO, LV.
    """
    stime = time.time()
    model.eval()
    metrics_list = []
    with torch.no_grad():
        for i, sample in enumerate(tqdm(test_loader)):
            img, msk = sample['image'], sample['label']
            # case_name = sample['patient_id'][0] if 'patient_id' in sample else f'case_{i}'
            patient_id = sample['patient_id'][0] if isinstance(sample['patient_id'], (list, tuple, torch.Tensor)) else \
            sample['patient_id']
            frame_num = sample['frame_num'][0] if isinstance(sample['frame_num'], (list, tuple, torch.Tensor)) else \
            sample['frame_num']
            case_name = f"{patient_id}_frame{frame_num}"
            metrics, pred = test_single_volume_acdc(
                img, msk, model, classes=config.num_classes,
                patch_size=[config.input_size_h, config.input_size_w],
                test_save_path=test_save_path, case=case_name, z_spacing=config.z_spacing,
                save_slices_dir=save_slices_dir
            )
            metrics_list.append(list(metrics.values()))
            logger.info(
                f'Case {case_name}: DSC={metrics["DSC"]:.4f} RV={metrics["RV"]:.4f} MYO={metrics["MYO"]:.4f} LV={metrics["LV"]:.4f}'
            )
    metrics_arr = np.array(metrics_list)
    avg_metrics = metrics_arr.mean(axis=0)
    etime = time.time()
    log_info = (
        f'Test finished, Mean DSC: {avg_metrics[0]:.4f} (RV:{avg_metrics[1]:.4f}, MYO:{avg_metrics[2]:.4f}, LV:{avg_metrics[3]:.4f}), time(s): {etime-stime:.2f}'
    )
    print(log_info)
    logger.info(log_info)
    return avg_metrics