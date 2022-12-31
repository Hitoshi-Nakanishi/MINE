import time
from .util import AverageMeter, accuracy

def train(train_loader, model, criterion, optimizer, epoch, writer, device, args):
    """
    Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    kld_meter = AverageMeter()
    top1 = AverageMeter()
    forward_time = AverageMeter()
    kl_time = AverageMeter()
    backward_time = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    start_iter = len(train_loader)*epoch
    kl_fac = args.kl_fac if not args.no_ib else 0
    print('kl fac:{}'.format((kl_fac)))
    for i, (input, target) in enumerate(train_loader):
        ite = start_iter + i
        # measure data loading time
        data_time.update(time.time() - end)
        input, target = input.to(device), target.to(device)
        compute_start = time.time()
        if args.no_ib:
            output = model(input)
        else:
            output, kl_total = model(input)
            writer.add_scalar('train_kld', kl_total.data, ite)
        forward_time.update(time.time() - compute_start)
        ce_loss = criterion(output, target)
        
        loss = ce_loss
        if kl_fac > 0:
            loss += kl_total * kl_fac

        # compute gradient and do SGD step
        optimizer.zero_grad()
        compute_start = time.time()
        loss.backward()
        backward_time.update(time.time()-compute_start)
        optimizer.step()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(ce_loss.item(), input.size(0))
        kld_meter.update(kl_total.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Date: {date}\t'
                  'Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Forward Time {forward_time.val:.3f} ({forward_time.avg:.3f})\t'
                  'KL Time {kl_time.val:.3f} ({kl_time.avg:.3f})\t'
                  'Backward Time {backward_time.val:.3f} ({backward_time.avg:.3f})\t'
                  'CE {loss.val:.4f} ({loss.avg:.4f})\t'
                  'KLD {klds.val:.4f} ({klds.avg:.4f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), date=time.strftime("%Y-%m-%d %H:%M:%S"), batch_time=batch_time,
                      forward_time=forward_time, backward_time=backward_time, kl_time=kl_time,
                      data_time=data_time, loss=losses, klds=kld_meter, top1=top1))
    print('Date: {date}\t'
        'Epoch: [{0}][{1}/{2}]\t'
        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        'Forward Time {forward_time.val:.3f} ({forward_time.avg:.3f})\t'
        'KL Time {kl_time.val:.3f} ({kl_time.avg:.3f})\t'
        'Backward Time {backward_time.val:.3f} ({backward_time.avg:.3f})\t'
        'CE {loss.val:.4f} ({loss.avg:.4f})\t'
        'KLD {klds.val:.4f} ({klds.avg:.4f})\t'
        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
            epoch, i, len(train_loader), date=time.strftime("%Y-%m-%d %H:%M:%S"), batch_time=batch_time,
            forward_time=forward_time, backward_time=backward_time, kl_time=kl_time,
            data_time=data_time, loss=losses, klds=kld_meter, top1=top1))
    writer.add_scalar('train_ce_loss', losses.avg, epoch)