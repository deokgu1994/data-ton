import argparse
from gc import collect
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.model as module_arch
from parse_config import ConfigParser

import model.model as module_net
import optimizer.metric as module_metric
import data_loader.data_sets as module_data_set

def main(config):
    logger = config.get_logger('test')


    # setup data_loader instances
    data_set = config.init_obj("data_set", module_data_set)
    if config["debug"]["set_debug"]:
        data_set.ratio = config["debug"]["ratio"]
    valid_data_loader = DataLoader(data_set,
                    batch_size= config["batch_size"],
                    shuffle= config["shuffle"],
                    num_workers= config["num_workers"],
                    drop_last=True 
                )

    # build model architecture
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = {}
    for i in range(config["Net_num"]):
        model[i] = {"model": config.init_obj(f'Net{i+1}', module_arch), "path": config[f'Net{i+1}']["pth_path"]}
        checkpoint = torch.load(model[i]["path"])
        # if config['n_gpu'] > 1:
        #     model = torch.nn.DataParallel(model[i]["model"])
        model[i]["model"].load_state_dict(checkpoint)
        model[i]["model"] = model[i]["model"].to(device)
        model[i]["model"].eval()

    # get function handles of loss and metrics
    # loss_fn = getattr(module_loss, config['loss'])
    # metric_fns = [getattr(module_metric, met) for met in config['metrics']]



    # total_metrics = torch.zeros(len(metric_fns))
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(valid_data_loader)):
            data, target = data.to(device), target.to(device)
            for _key in model:
            
                output = model(data)

            #
            # save sample images, or do something with output here
            #

            # computing loss, metrics on test set
            # loss = loss_fn(output, target)
            batch_size = data.shape[0]
            # total_loss += loss.item() * batch_size
    #         for i, metric in enumerate(metric_fns):
    #             total_metrics[i] += metric(output, target) * batch_size

    # n_samples = len(data_loader.sampler)
    # log = {'loss': total_loss / n_samples}
    # log.update({
    #     met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    # })
    # logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
