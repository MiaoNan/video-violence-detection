"""Training orchestration; dataset and epoch implementations are external."""
import torch
from torch import nn
from epoch import train, val
from utils import Log
from models.factory import build_model
from training.data import build_loaders
from run_output import create_run_dir


def main(config):
    run_dir = create_run_dir(f"train_{config.model}_{config.dataset}_cv{config.num_cv}")
    config.run_dir = str(run_dir)
    log_dir = run_dir / "logs"
    weights_dir = run_dir / "weights"
    log_dir.mkdir()
    weights_dir.mkdir()
    print(f"Run output: {run_dir}")
    device = config.device
    model, params = build_model(config)
    train_loader, val_loader = build_loaders(config)
    dataset = config.dataset
    sample_duration = config.sample_duration
    cv = config.num_cv

    # log
    batch_log = Log(
        str(log_dir / '{}_fps{}_{}_batch{}.log'.format(
            config.model,
            sample_duration,
            dataset,
            cv,
        )), ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])
    epoch_log = Log(
        str(log_dir / '{}_fps{}_{}_epoch{}.log'.format(config.model, sample_duration,
                                               dataset, cv)),
        ['epoch', 'loss', 'acc', 'lr'])
    val_log = Log(
        str(log_dir / '{}_fps{}_{}_val{}.log'.format(config.model, sample_duration,
                                             dataset, cv)),
        ['epoch', 'loss', 'acc'])

    # prepare
    criterion = nn.CrossEntropyLoss().to(device)

    learning_rate = config.learning_rate
    momentum = config.momentum
    weight_decay = config.weight_decay

    optimizer = torch.optim.SGD(params=params,
                                lr=learning_rate,
                                momentum=momentum,
                                weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           factor=config.factor,
                                                           min_lr=config.min_lr)

    acc_baseline = float("-inf")
    loss_baseline = float("inf")

    for i in range(config.num_epoch):
        train(i, train_loader, model, criterion, optimizer, device, batch_log,
              epoch_log)
        val_loss, val_acc = val(i, val_loader, model, criterion, device,
                                val_log)
        scheduler.step(val_loss)
        if val_acc > acc_baseline or (val_acc >= acc_baseline and
                                      val_loss < loss_baseline):
            torch.save(
                model.state_dict(),
                str(weights_dir / '{}_fps{}_{}{}_{}_{:.4f}_{:.6f}.pth'.format(
                    config.model, sample_duration, dataset, cv, i, val_acc,
                    val_loss)))
            acc_baseline = val_acc
            loss_baseline = val_loss

    return run_dir
