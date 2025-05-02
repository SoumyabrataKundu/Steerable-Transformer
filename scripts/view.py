import torch
import torch.nn as nn
import torch.utils
import torch.backends.cudnn
from torch.utils.data import DataLoader
import numpy as np
import os
import logging
import copy
import sys
import h5py
import time

from Steerable.Segmentation.loss import SegmentationLoss
from Steerable.Segmentation.metrics import Metrics


def main(model_path, data_path, batch_size, n_radius, max_m, loss_type):
   
 ################################################################################################################################### 
 ##################################################### Logging #####################################################################
 ###################################################################################################################################
    arguments = copy.deepcopy(locals())
    
    log_dir = os.path.join(model_path, 'log/')
    log_file = 'r' + str(n_radius) + 'k' + str(max_m)
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    
    # Creating the logger
    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    ch = logging.StreamHandler()
    logger.addHandler(ch)
    fh = logging.FileHandler(os.path.join(log_dir, "log." + log_file + ".txt"), mode = "w")
    logger.addHandler(fh)

    logger.info("%s", repr(arguments))
    logger.info("\n\n")
    torch.backends.cudnn.benchmark = True
    
 ###################################################################################################################################
 ################################## Loading Model, Datasets, Loss and Optimizer ####################################################
 ###################################################################################################################################
    
    # Load the model
    model = Model(n_radius, max_m)
    num_classes = model.num_classes
    device = torch.device("cuda")
    model = nn.DataParallel(model)
    model = model.to(device)
    
    logger.info("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    
    # Datasets
    datasets = get_datasets(data_path)
    ## Dataloaders
    test_loader = torch.utils.data.DataLoader(datasets['test'], batch_size = batch_size)
    # Loss
    criterion = SegmentationLoss(loss_type=loss_type)


    ########################################################################################################################
    ############################################### Test Function ##########################################################
    ########################################################################################################################

    def test(inputs, labels):
        inputs = inputs.to(device)
        labels = labels.to(device)
  
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels).item()
            preds = torch.argmax(outputs, dim=1)

            metrics.add_to_confusion_matrix(preds, labels)
            

        return outputs, loss


    ###########################################################################################################################
    ################################################### Testing Loop ##########################################################
    ###########################################################################################################################
    
    total_loss = 0
    model.eval()
    metrics = Metrics(num_classes)
    f = h5py.File(os.path.join(log_dir, 'outputs.hdf5'), 'w')
    prob_dataset = None

    if datasets['val'] is not None:
        model.load_state_dict(torch.load(os.path.join(model_path, 'log', 'best_state.'+ log_file + '.pkl')))
    else:
        model.load_state_dict(torch.load(os.path.join(model_path, 'log', 'state.'+ log_file + '.pkl')))
    
    
    logger.info(f"\n\n\nTesting:\n")

    
    for batch_index, (inputs, labels) in enumerate(test_loader):

        t0 = time.time()
        outputs, current_loss = test(inputs, labels)
        t1 = time.time()

        if prob_dataset is None:
            f.create_dataset('probs', (0, ) + outputs.shape[1:], maxshape=(None,) +  outputs.shape[1:], chunks=True)
            prob_dataset = f['probs']
        
        prob_dataset.resize((len(prob_dataset) + len(outputs),) + outputs.shape[1:])
        prob_dataset[-len(outputs):] = outputs.cpu()
        total_loss += current_loss * len(inputs)

        logger.info(f'Test [{batch_index+1}/{len(test_loader)}] Time : {(t1-t0)*1e3:.1f} ms  Loss={current_loss:.2f} \t Metrics={torch.tensor(metrics.all_metrics())}')
       
    f.close()
    avg_loss = total_loss / len(datasets['test'])
    logger.info(f'\n\nOverall Loss = {avg_loss:.4f}')

    print(f'\n\nTesting Loss = {avg_loss:.4f}')
    print(f'\n\nTesting Metrics :')
    print(f'\nMean IOU = {metrics.mIOU():.4f}')
    print(f'Freq IOU = {metrics.fIOU():.4f}')
    print(f'Pixel Acc = {metrics.pixel_accuracy():.4f}')
    print(f'Mean Acc = {metrics.mean_accuracy():.4f}')
    print(f'Mean Dice = {metrics.mDice():.4f}')
    print(f'Freq Dice = {metrics.fDice():.4f}')
    print(f'\nIOU per class :\n{metrics.iou_per_class()}')
    print(f'\nDice per class:\n{metrics.dice_per_class()}')



############################################################################################################################
################################################### Argument Parser ########################################################
############################################################################################################################


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--n_radius", type=int,required=True)
    parser.add_argument("--max_m", type=int, required=True)
    parser.add_argument("--loss_type", type=str, required=True)

    args = parser.parse_args()
    sys.path.append(args.__dict__['model_path'])
    from model import Model, get_datasets

    main(**args.__dict__)

