'''
Descripttion: 
version: 1.0
Author: Areebol
Date: 2023-07-19 20:35:06
'''
'''
Descripttion: 
version: 1.0
Author: Areebol
Date: 2023-07-19 20:35:06
'''

# Import config
from parser_config import args
from utils.calibration_library.visualization import ReliabilityDiagram, ConfidenceHistogram, ReliabilityDiagramPerClass
from utils.calibration_library.metrics import ECELoss, SCELoss
from utils.calibration_library.utils import AverageMeter,accuracy,validate
from pytorch_lightning.callbacks import ModelCheckpoint
import utils.utils as utils
import torch
import time
import torch.nn as nn
import model
import warmup_scheduler
import numpy as np
import pytorch_lightning as pl
import torchvision

# Get dataset 
train_dl, test_dl = utils.get_dataloader(args=args)

# Get training model
class Net(pl.LightningModule):
    def __init__(self, hparams, test_dl):
        super(Net, self).__init__()
        self.hparams.update(vars(hparams))
        self.model = model.get_model(hparams)
        self.criterion = utils.get_criterion(args)
        self.test_dl = test_dl
        if hparams.cutmix:
            self.cutmix = utils.CutMix(hparams.size, beta=1.)
        if hparams.mixup:
            self.mixup = utils.MixUp(alpha=1.)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr, betas=(
            self.hparams.beta1, self.hparams.beta2), weight_decay=self.hparams.weight_decay)
        self.base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.min_lr)
        self.scheduler = warmup_scheduler.GradualWarmupScheduler(
            self.optimizer, multiplier=1., total_epoch=self.hparams.warmup_epoch, after_scheduler=self.base_scheduler)
        return [self.optimizer], [self.scheduler]

    def training_step(self, batch, batch_idx):
        img, label = batch
        if self.hparams.cutmix or self.hparams.mixup:
            if self.hparams.cutmix:
                img, label, rand_label, lambda_ = self.cutmix((img, label))
            elif self.hparams.mixup:
                if np.random.rand() <= 0.8:
                    img, label, rand_label, lambda_ = self.mixup((img, label))
                else:
                    img, label, rand_label, lambda_ = img, label, torch.zeros_like(
                        label), 1.
            out = self.model(img)
            loss = self.criterion(out, label)*lambda_ + \
                self.criterion(out, rand_label)*(1.-lambda_)
        else:
            out = self(img)
            loss = self.criterion(out, label)

        acc = torch.eq(out.argmax(-1), label).float().mean()
        self.log("loss", loss)
        self.log("acc", acc)
        return loss

    def training_epoch_end(self, outputs):
        self.log(
            "lr", self.optimizer.param_groups[0]["lr"], on_epoch=self.current_epoch)
        prec1, ece_loss, sce_loss = validate(
            args, self.test_dl, self.model, self.criterion, None)
        self.log("prec1", prec1)
        self.log("ece_loss", ece_loss)
        self.log("sce_loss", sce_loss)

    def validation_step(self, batch, batch_idx):
        img, label = batch
        out = self(img)
        loss = self.criterion(out, label)
        acc = torch.eq(out.argmax(-1), label).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return loss


if __name__ == "__main__":
    experiment_name = utils.get_experiment_name(args)
    logger = pl.loggers.TensorBoardLogger(
        save_dir="logs",
        name=experiment_name
    )
    model_path = f"weights/{experiment_name}/"

    # 监视测试准确率，注意这里性能指标名字要与self.log中保持一致
    ckpt_callback_test_acc = ModelCheckpoint(
        monitor='val_acc', dirpath=model_path,
        filename='epoch{epoch:02d}-val_acc{Val_accuracy:.2f}',
        save_last=True, save_weights_only=True, mode='max',
        save_top_k=5)

    refresh_rate = 1

    net = Net(args, test_dl=test_dl)
    trainer = pl.Trainer(precision=args.precision, fast_dev_run=args.dry_run,
                         gpus=[args.gpu], benchmark=args.benchmark,
                         logger=logger, max_epochs=args.max_epochs,
                         weights_summary="full", progress_bar_refresh_rate=refresh_rate,
                         callbacks=[ckpt_callback_test_acc])
    trainer.fit(model=net, train_dataloader=train_dl, val_dataloaders=test_dl)
    if not args.dry_run:
        model_path = f"weights/{experiment_name}.pth"
        torch.save(net.state_dict(), model_path)
        
