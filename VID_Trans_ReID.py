import argparse
import os
import random
import time

import numpy as np
import torch
from torch.cuda import amp
from torch_ema import ExponentialMovingAverage

from Dataloader import dataloader
from Loss_fun import make_loss
from VID_Test import test
from VID_Trans_model import VID_Trans
from utility import AverageMeter, optimizer as build_optimizer, scheduler as build_scheduler


CENTER_LOSS_WEIGHT = 0.0005


def set_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VID-Trans-ReID camera-removed baseline training')
    parser.add_argument('--Dataset_name', required=True, help='Dataset name', type=str)
    parser.add_argument('--model_path', required=True, help='ViT pretrained weight path', type=str)
    parser.add_argument('--output_dir', default='./output_camera_removed', help='Checkpoint output directory', type=str)
    parser.add_argument('--epochs', default=120, help='Number of epochs', type=int)
    parser.add_argument('--eval_every', default=10, help='Evaluate every N epochs', type=int)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(1234)

    train_loader, _, num_classes, camera_num, _, q_val_set, g_val_set = dataloader(args.Dataset_name)
    model = VID_Trans(num_classes=num_classes, camera_num=camera_num, pretrainpath=args.model_path)

    loss_fun, center_criterion_global, center_criterion_local = make_loss(num_classes=num_classes)
    optimizer_center_global = torch.optim.SGD(center_criterion_global.parameters(), lr=0.5)
    optimizer_center_local = torch.optim.SGD(center_criterion_local.parameters(), lr=0.5)

    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer)
    scaler = amp.GradScaler()

    device = 'cuda'
    model = model.to(device)
    center_criterion_global = center_criterion_global.to(device)
    center_criterion_local = center_criterion_local.to(device)

    ema = ExponentialMovingAverage(model.parameters(), decay=0.995)
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    best_rank1 = 0.0

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        scheduler.step(epoch)
        model.train()

        for iteration, (img, pid, _target_cam, labels2) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            optimizer_center_global.zero_grad()
            optimizer_center_local.zero_grad()

            img = img.to(device)
            pid = pid.to(device)
            labels2 = labels2.to(device)

            with amp.autocast(enabled=True):
                score, feat, a_vals = model(img, pid)
                attn_noise = a_vals * labels2
                attn_loss = attn_noise.sum(1).mean()
                loss_id, center_loss = loss_fun(score, feat, pid)
                loss = loss_id + CENTER_LOSS_WEIGHT * center_loss + attn_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            ema.update()

            scale_back = 1.0 / CENTER_LOSS_WEIGHT
            for criterion in [center_criterion_global, center_criterion_local]:
                for param in criterion.parameters():
                    if param.grad is not None:
                        param.grad.data *= scale_back

            scaler.step(optimizer_center_global)
            scaler.step(optimizer_center_local)
            scaler.update()

            if isinstance(score, list):
                acc = (score[0].max(1)[1] == pid).float().mean()
            else:
                acc = (score.max(1)[1] == pid).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc.item(), 1)

            torch.cuda.synchronize()
            if iteration % 50 == 0:
                print(
                    'Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}'.format(
                        epoch, iteration, len(train_loader), loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]
                    )
                )

        epoch_time = time.time() - start_time
        print('Epoch {} finished in {:.1f}s'.format(epoch, epoch_time))

        if epoch % args.eval_every == 0:
            model.eval()
            rank1, mAP = test(model, q_val_set, g_val_set)
            print('CMC: %.4f, mAP : %.4f' % (rank1, mAP))
            latest_path = os.path.join(args.output_dir, f'{args.Dataset_name}_camera_removed_latest.pth')
            torch.save(model.state_dict(), latest_path)
            if best_rank1 < rank1:
                best_rank1 = rank1
                best_path = os.path.join(args.output_dir, f'{args.Dataset_name}_camera_removed_best.pth')
                torch.save(model.state_dict(), best_path)
                print(f'[OK] Saved best checkpoint: {best_path}')
