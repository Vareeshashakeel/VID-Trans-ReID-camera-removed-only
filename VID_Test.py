import argparse
import numpy as np
import torch

from Dataloader import dataloader
from VID_Trans_model import VID_Trans


def evaluate(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=21):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print(f'Note: number of gallery samples is quite small, got {num_g}')

    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc = []
    all_ap = []
    num_valid_q = 0.0

    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.0

        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = np.asarray([x / (i + 1.0) for i, x in enumerate(tmp_cmc)]) * orig_cmc
        ap = tmp_cmc.sum() / num_rel
        all_ap.append(ap)

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_ap)

    return all_cmc, mAP


def _scalar_pid(pids):
    if isinstance(pids, (list, tuple)):
        pids = pids[0]
    if torch.is_tensor(pids):
        pids = pids.item()
    return int(pids)


def _scalar_camid(camids):
    if torch.is_tensor(camids):
        return int(camids.view(-1)[0].item())
    if isinstance(camids, (list, tuple)):
        first = camids[0]
        if torch.is_tensor(first):
            return int(first.item())
        return int(first)
    return int(camids)


def test(model, queryloader, galleryloader, pool='avg', use_gpu=True):
    model.eval()

    qf, q_pids, q_camids = [], [], []
    with torch.no_grad():
        for _, (imgs, pids, camids, _) in enumerate(queryloader):
            # imgs: [1, num_clips, seq_len, C, H, W]
            imgs = imgs.squeeze(0)  # [num_clips, seq_len, C, H, W]

            if use_gpu:
                imgs = imgs.cuda(non_blocking=True)

            num_clips = imgs.size(0)
            pid = _scalar_pid(pids)
            camid = _scalar_camid(camids)

            features = model(imgs, None)
            features = features.view(num_clips, -1)
            features = torch.mean(features, dim=0).cpu()

            qf.append(features)
            q_pids.append(pid)
            q_camids.append(camid)

        qf = torch.stack(qf, dim=0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print(f'Extracted features for query set, obtained {qf.size(0)}-by-{qf.size(1)} matrix')

        gf, g_pids, g_camids = [], [], []
        for _, (imgs, pids, camids, _) in enumerate(galleryloader):
            imgs = imgs.squeeze(0)  # [num_clips, seq_len, C, H, W]

            if use_gpu:
                imgs = imgs.cuda(non_blocking=True)

            num_clips = imgs.size(0)
            pid = _scalar_pid(pids)
            camid = _scalar_camid(camids)

            features = model(imgs, None)
            features = features.view(num_clips, -1)

            if pool == 'avg':
                features = torch.mean(features, dim=0)
            else:
                features, _ = torch.max(features, dim=0)

            features = features.cpu()

            gf.append(features)
            g_pids.append(pid)
            g_camids.append(camid)

    gf = torch.stack(gf, dim=0)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)

    print(f'Extracted features for gallery set, obtained {gf.size(0)}-by-{gf.size(1)} matrix')
    print('Computing distance matrix')

    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n)
    distmat = distmat + torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(qf, gf.t(), beta=1, alpha=-2)
    distmat = distmat.numpy()

    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    print('Results ----------')
    print('mAP: {:.1%}'.format(mAP))
    print('CMC curve')
    print('Rank-1  : {:.1%}'.format(cmc[0]))
    if len(cmc) > 4:
        print('Rank-5  : {:.1%}'.format(cmc[4]))
    if len(cmc) > 9:
        print('Rank-10 : {:.1%}'.format(cmc[9]))
    if len(cmc) > 19:
        print('Rank-20 : {:.1%}'.format(cmc[19]))

    return cmc[0], mAP


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VID-Trans-ReID camera-removed baseline test')
    parser.add_argument('--Dataset_name', required=True, help='Dataset name', type=str)
    parser.add_argument('--model_path', required=True, help='Trained model checkpoint path', type=str)
    args = parser.parse_args()

    _, _, num_classes, camera_num, _, q_val_loader, g_val_loader = dataloader(args.Dataset_name)

    model = VID_Trans(
        num_classes=num_classes,
        camera_num=camera_num,
        pretrainpath=None
    ).cuda()

    checkpoint = torch.load(args.model_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']

    model.load_state_dict(checkpoint, strict=True)

    cmc, mAP = test(model, q_val_loader, g_val_loader)
    print('CMC: %.4f, mAP : %.4f' % (cmc, mAP))
