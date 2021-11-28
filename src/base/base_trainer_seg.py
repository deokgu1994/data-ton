import torch
from torch.utils.data import DataLoader
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter
# from sklearn.model_selection import KFold, StratifiedKFold
# from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from utils import making_group, collate_fn
import copy


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, criterion, data_set, optimizer =None, config = None):
        self.config = config
        if self.config["save"] : self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.data_set = data_set

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.early_stop = cfg_trainer["early_stop"]
        # self.monitor = cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best
        # if self.monitor == 'off':
        #     self.mnt_mode = 'off'
        #     self.mnt_best = 0
        # else:
        #     self.mnt_mode, self.mnt_metric = self.monitor.split()
        #     assert self.mnt_mode in ['min', 'max']

        #     self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        #     self.early_stop = cfg_trainer.get('early_stop', inf)
        #     if self.early_stop <= 0:
        #         self.early_stop = inf

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance                
        # self.writer = TensorboardWriter(config.log_dir, None, cfg_trainer['tensorboard'])

        # if config.resume is not None:
        #     self._resume_checkpoint(config.resume)


    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        
        # FIXME: 설정에 따라서 일반, k, strati, multilabel 설정하자.
        Y = making_group(self.data_set)
        mskf = MultilabelStratifiedKFold(n_splits=self.config["fold_split"],  shuffle=True, random_state=self.config["seed"])

        for kfold, (train_index, validate_index) in enumerate(mskf.split(self.data_set, Y)):
            train_dataset = torch.utils.data.dataset.Subset(self.data_set, train_index)
            valid_dataset = torch.utils.data.dataset.Subset(self.data_set, validate_index)

            train_loader = DataLoader(train_dataset,
                            batch_size=self.config["batch_size"],
                            shuffle=self.config["shuffle"],
                            num_workers=self.config["num_workers"],
                            collate_fn=collate_fn,
                            drop_last=True 
                        )

            val_loader = DataLoader(valid_dataset,
                            batch_size=self.config["batch_size"],
                            shuffle=self.config["shuffle"],
                            num_workers=self.config["num_workers"],
                            collate_fn=collate_fn,
                            drop_last=True 
                        )
            not_improved_count = 0 
            kfold += 1
            best_mIoU = 0
            self.model.init_weights()
            for epoch in range(self.start_epoch, self.epochs + 1):
                result = self._train_epoch(epoch, kfold, train_loader, val_loader)

                # save logged informations into log dict
                log = {'epoch': epoch, "kfold": kfold}
                log.update(result)

                # print logged informations to the screen
                for key, value in log.items():
                    if self.config["save"] : self.logger.info('{:15s}: {}'.format(str(key), value))

                # evaluate model performance according to configured metric, save best checkpoint as model_best
                best = False
                # if self.mnt_mode != 'off':
                    # try:
                    #     # check whether model performance improved or not, according to specified metric(mnt_metric)
                    #     improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                    #             (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                    # except KeyError:
                    #     if self.config["save"] : self.logger.warning("Warning: Metric '{}' is not found. "
                    #                         "Model performance monitoring is disabled.".format(self.mnt_metric))
                    #     self.mnt_mode = 'off'
                    #     improved = False
                if log["val_mIoU"] > best_mIoU:
                    improved = True
                    best_mIoU = log["val_mIoU"]
                else:
                    improved = False

                if improved:
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    if self.config["save"] :self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                    "Training stops.".format(self.early_stop))
                    break

                if epoch % self.save_period == 0 or best:
                    self._save_checkpoint(epoch, kfold, save_best = best)

    def _save_checkpoint(self, epoch, kfold, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        if self.config["save"] :
            if save_best:
                filename = str(self.checkpoint_dir / 'model_best-kfold{}-epoch{}.pth'.format(kfold, epoch))
                torch.save(self.model.state_dict(), filename)
                self.logger.info("Saving current best: model_best.pth ...")
            else:
                filename = str(self.checkpoint_dir / 'checkpoint-kfold{}-epoch{}.pth'.format(kfold,epoch))
                torch.save(self.model.state_dict(), filename)
                self.logger.info("Saving checkpoint: {} ...".format(filename))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

def collate_fn(batch):
    return tuple(zip(*batch))
