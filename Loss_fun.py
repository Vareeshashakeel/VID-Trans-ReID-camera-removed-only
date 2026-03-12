from loss.center_loss import CenterLoss
from loss.softmax_loss import CrossEntropyLabelSmooth
from loss.triplet_loss import TripletLoss


def make_loss(num_classes):
    center_global = CenterLoss(num_classes=num_classes, feat_dim=768, use_gpu=True)
    center_local = CenterLoss(num_classes=num_classes, feat_dim=3072, use_gpu=True)

    triplet = TripletLoss()
    xent = CrossEntropyLabelSmooth(num_classes=num_classes)

    def loss_func(score, feat, target):
        if isinstance(score, list):
            id_loss_local = [xent(scor, target) for scor in score[1:]]
            id_loss_local = sum(id_loss_local) / len(id_loss_local)
            id_loss = 0.25 * id_loss_local + 0.75 * xent(score[0], target)
        else:
            id_loss = xent(score, target)

        if isinstance(feat, list):
            tri_loss_local = [triplet(feats, target)[0] for feats in feat[1:]]
            tri_loss_local = sum(tri_loss_local) / len(tri_loss_local)
            tri_loss = 0.25 * tri_loss_local + 0.75 * triplet(feat[0], target)[0]

            center_loss_global = center_global(feat[0], target)
            center_loss_local = [center_local(feats, target) for feats in feat[1:]]
            center_loss_local = sum(center_loss_local) / len(center_loss_local)
            center_loss = 0.75 * center_loss_global + 0.25 * center_loss_local
        else:
            tri_loss = triplet(feat, target)[0]
            center_loss = center_global(feat, target)

        return id_loss + tri_loss, center_loss

    return loss_func, center_global, center_local
