from utils.calibration_library.visualization import ReliabilityDiagram, ConfidenceHistogram, ReliabilityDiagramPerClass
from utils.calibration_library.metrics import ECELoss, SCELoss
import time
import torch
import torch.nn as nn
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def validate(args, val_loader, model, criterion, model_name):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    mean_confidence = AverageMeter()
    mean_error_confidence = AverageMeter()
    ece_criterion = ECELoss()
    sce_criterion = SCELoss()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    outputs_list, labels_list = [], []
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            if args.half:
                input_var = input_var.half()

            if args.model_name == 'shuffle_vit':
                # compute output
                output = model(input_var, is_shuffle=False) / args.T
            else:
                output = model(input_var) / args.T

            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # 记录所有的softmax输出
            softmax_output = nn.Softmax(dim=1)(output)
            outputs_list.append(softmax_output)
            labels_list.append(target_var)

            # 计算平均置信度
            max_prob = torch.max(softmax_output, dim=1).values
            batch_mean_confidence = torch.mean(max_prob)
            mean_confidence.update(batch_mean_confidence, input.size(0))

            # 计算错误样本的平均置信度
            output_indices = torch.max(softmax_output, dim=1).indices
            error_max_prob = max_prob[output_indices != target_var]
            batch_mean_error_confidence = torch.mean(error_max_prob)
            mean_error_confidence.update(
                batch_mean_error_confidence, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    outputs = torch.cat(outputs_list).cpu().numpy()
    labels = torch.cat(labels_list).cpu().numpy()

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))
    print(f' * avg_confidence {mean_confidence.avg:.3f}')
    print(f' * avg_error_confidence {mean_error_confidence.avg:.3f}')

    ece_loss = ece_criterion.loss(outputs, labels, logits=False)
    print(f' * ece_loss {ece_loss:.3f}')

    sce_loss = sce_criterion.loss(outputs, labels, logits=False)
    print(f' * sce_loss {sce_loss:.3f}')

    return top1.avg, ece_loss, sce_loss
