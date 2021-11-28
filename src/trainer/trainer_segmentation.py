import numpy as np
import torch
from torchvision.utils import make_grid
from base.base_trainer_seg import BaseTrainer
from utils import inf_loop, MetricTracker
from tqdm import tqdm
from utils import label_accuracy_score, add_hist


class Trainer_seg(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, optimizer, config, device,
                 data_set, lr_scheduler=None, len_epoch=None):
        super().__init__(model = model, criterion = criterion,  data_set = data_set, optimizer = optimizer, config = config)
        self.config = config
        self.device = device
        self.data_set = data_set

        self.lr_scheduler = lr_scheduler
        self.do_validation = True

        self.log_step = self.config["log_step"]

    def _train_epoch(self, epoch, kfold, train_loader, val_loader):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.len_epoch = len(train_loader)
        # print(self.len_epoch)

        train_loader.dataset.mode = "train"
        # self.train_metrics.reset()
        hist = np.zeros((11, 11))
        for batch_idx, (images, masks, _) in enumerate(train_loader):
            images = torch.stack(images)       
            masks = torch.stack(masks).long() 
            
            # gpu 연산을 위해 device 할당
            images, masks = images.to(self.device), masks.to(self.device)
            
            # inference
            outputs = self.model(images)
            
            # loss 계산 (cross entropy loss)
            loss = self.criterion(outputs, masks)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            
            hist = add_hist(hist, masks, outputs, n_class=11)
            acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
            
            # step 주기에 따른 loss 출력
            if batch_idx % self.log_step == 0 :
                if self.config["save"] : self.logger.debug('Train kfold: {} Epoch: {} {} Loss: {:.6f} mIOU: {:.4f}'.format(
                    kfold,
                    epoch,
                    self._progress(batch_idx, train_loader),
                    loss.item(),
                    mIoU))

            if batch_idx == self.len_epoch:
                break

        log = {"acc": acc,"acc_cls": acc_cls, "mIoU" : mIoU, "fwavacc": fwavacc,}
        if self.do_validation:
            val_log = self._valid_epoch(epoch, kfold, val_loader)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch, kfold, val_loader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        val_loader.dataset.mode = "val"
        # self.valid_metrics.reset()
        with torch.no_grad():
            n_class = 11
            total_loss = 0
            cnt = 0
            
            hist = np.zeros((11, 11))
            for step, (images, masks, _) in enumerate(val_loader):
                images = torch.stack(images)       
                masks = torch.stack(masks).long()  

                images, masks = images.to(self.device), masks.to(self.device)            
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                total_loss += loss
                cnt += 1
                
                outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                masks = masks.detach().cpu().numpy()
                
                hist = add_hist(hist, masks, outputs, n_class=11)
            
            acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
            IoU_by_class = [{classes : round(IoU,4)} for IoU, classes in zip(IoU , ['Backgroud', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing'])]
            
            avrg_loss = total_loss / cnt
            if self.config["save"] : self.logger.debug('Train kfold: {} Epoch: {} Loss: {:.6f} mIOU: {:.4f}'.format(
                    kfold,
                    epoch,
                    avrg_loss.item(),
                    mIoU))
            if self.config["save"] : self.logger.debug(f'IoU by class : {IoU_by_class}')

            log = {"acc": acc,"acc_cls": acc_cls, "mIoU" : mIoU, "fwavacc": fwavacc, "IoU_by_class": IoU_by_class}
        return log

    def _progress(self, batch_idx, loader):
        base = '[{}/{} ({:.0f}%)]'
        # if hasattr(self.data_loader, 'n_samples'):
        #     current = batch_idx * self.data_loader.batch_size
        #     total = self.data_loader.n_samples
        # else:
        current = batch_idx
        total = len(loader)
        return base.format(current, total, 100.0 * current / total)
