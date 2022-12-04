from lib.utils import *
import torch
import sys
import numpy as np
import time
import wandb


def validate(rank, val_loader, model, criterion, configs, logger):
    # Mean/Std for normalization
    mean = torch.Tensor(np.array(configs.TRAIN.mean)[:, np.newaxis, np.newaxis])
    mean = mean.expand(3, 224, 224).cuda()
    std = torch.Tensor(np.array(configs.TRAIN.std)[:, np.newaxis, np.newaxis])
    std = std.expand(3, 224, 224).cuda()

    # Initiate the meters
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to evaluate mode
    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        with torch.no_grad():
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            input = input - mean
            input.div_(std)
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if rank == 0 and i % configs.TRAIN.print_freq == 0:
                print(
                    "Test: [{0}/{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                    "Prec@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                        i,
                        len(val_loader),
                        batch_time=batch_time,
                        loss=losses,
                        top1=top1,
                        top5=top5,
                    )
                )
                sys.stdout.flush()
        # break #@debug

    if rank == 0:
        print(
            " Final Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}".format(
                top1=top1, top5=top5
            )
        )

    return top1.avg
