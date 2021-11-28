import numpy as np
import torch
from torchvision.utils import make_grid
from base.base_trainer_cls import BaseTrainer
from utils import inf_loop, MetricTracker, inf_loop, MetricTracker, mixup_data, mixs_criterion, cutmix_data
from tqdm import tqdm
import wandb
import loss.loss as module_loss
from adamp import AdamP

class Trainer_cls(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,# data_loader, valid_data_loader=None, 
                data_set  = None, lr_scheduler=None, len_epoch=None, cut_mix = False, beta= 0.8, mix_up= False):
        super().__init__(model, criterion, data_set, device,metric_ftns, optimizer, config)
        self.config = config
        self.device = device

        self.do_validation = True
        self.data_set = data_set

        self.cut_mix = cut_mix 
        self.beta = beta
        self.mix_up = mix_up
 

        self.lr_scheduler = lr_scheduler

        self.log_step = self.config["log_step"]

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

        # -- logging
        wandb.init(project='data-ton', entity='cv-09-segmentation', name =f'{self.config["Net"]["args"]["model_name"]}')
        wandb.config = config
        wandb.watch(model)

    def _train_epoch(self, epoch, kfold, train_loader, val_loader):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        train_loader.dataset.mode = "train"
        target_count = 0
        len_epoch = len(train_loader)
        self.criterion = module_loss.LabelSmoothing()
        self.optimizer = AdamP(self.model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay= 1e-2)
        out_list = []
        target_list = []
        for batch_idx, (data, target) in enumerate(train_loader):
            # data, target = data.type(torch.FloatTensor).to(self.device), target.to(self.device)
            data, target = data.to(self.device), target.to(self.device)
            target_count+=len(target)
            self.optimizer.zero_grad()

            rand_num = np.random.random_integers(3) # 같이 넣어야 할까?
            if self.cut_mix and rand_num==1: # cutmix가 실행될 경우 
                data, target_a, target_b, lam = cutmix_data(data, target, self.beta, self.device)
                outputs = self.model(data)
                loss= mixs_criterion(self.criterion, outputs, target_a, target_b, lam)
            elif self.mix_up and rand_num ==2:
                data, target_a, target_b, lam = mixup_data(data, target)
                outputs = self.model(data)
                loss= mixs_criterion(self.criterion, outputs, target_a, target_b, lam)
            else:
                outputs= self.model(data) 
                loss= self.criterion(outputs, target) 

            # Backpropagation
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            out_list.extend(outputs)
            target_list.extend(target)

            if batch_idx % self.log_step == 0:
                if self.config["save"] : self.logger.debug('Train kfold: {} Epoch: {} {} Loss: {:.6f}'.format(
                    kfold,
                    epoch,
                    self._progress(batch_idx, train_loader),
                    loss.item()))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == len_epoch:
                break
            
        for met in self.metric_ftns:
            self.train_metrics.update(met.__name__, met(out_list, target_list))
        wandb.log({"train_loss":round(loss.item(), 4),**{k: v for k, v in self.train_metrics.result().items()}})
        output = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        images = wandb.Image(data, caption=f"Train - Target: {target}, Pred: {output}")          
        wandb.log({"predictions" : images})

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch, kfold, val_loader)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step(loss)

        return log

    def _valid_epoch(self, epoch, kfold, val_loader):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        val_loader.dataset.mode = "val"
        target_count = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.to(self.device), target.to(self.device)
                target_count += len(target)
                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(val_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
                wandb.log({"valid_loss":round(loss.item(), 4),**{'val_'+k: v for k, v in self.train_metrics.result().items()}})
       
            output = torch.argmax(output, dim=1).detach().cpu().numpy()
            images = wandb.Image(data, caption=f"Valid - Target: {target}, Pred: {output}")          
            wandb.log({"predictions" : images})
            
        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx,loader):
        base = '[{}/{} ({:.0f}%)]'
        current = batch_idx
        total = len(loader)
        return base.format(current, total, 100.0 * current / total)
