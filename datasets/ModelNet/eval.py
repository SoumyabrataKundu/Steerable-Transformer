import torch
import os
import numpy as np
import logging
import copy
import sys
import time
import pickle



def main(model_path, data_path, batch_size, n_radius, max_m, interpolation, restricted):

 ###################################################################################################################################
 ##################################################### Logging #####################################################################
 ###################################################################################################################################
    arguments = copy.deepcopy(locals())

    log_dir = os.path.join(model_path, 'log/')
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    # Creating the logger
    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    ch = logging.StreamHandler()
    logger.addHandler(ch)
    fh = logging.FileHandler(os.path.join(log_dir, "log.txt"), mode = "a")
    logger.addHandler(fh)

    logger.info("%s", repr(arguments))
    logger.info("\n\n")
    torch.backends.cudnn.benchmark = True
    
 ###################################################################################################################################
 ###################################################### Loading Model ##############################################################
 ###################################################################################################################################

    # Load the model
    model = Model(n_radius, max_m, interpolation, restricted)
    device = torch.device("cuda")
    model = model.to(device)
    model.eval()
    num_classes = model.network[-1].__dict__['out_features']

    try:
        model.load_state_dict(torch.load(os.path.join(model_path, 'log', 'best_state.pkl')))
    except:
        model.load_state_dict(torch.load(os.path.join(model_path, 'log', 'state.pkl')))


########################################################################################################################
############################################### Test Function ##########################################################
########################################################################################################################

    def test(inputs, labels):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs).reshape(-1, orientations, num_classes).mean(dim=1)
        labels = labels.reshape(-1, orientations)[:,0]
        _, predictions = torch.max(outputs, 1)

        n_correct_classwise = torch.zeros(num_classes).to(device)
        n_samples_classwise = torch.zeros(num_classes).to(device)

        for truth, pred in zip(labels, predictions):
            n_correct_classwise[truth] += truth == pred
            n_samples_classwise[truth] += 1

        return n_correct_classwise, n_samples_classwise
    

############################################################################################################################
################################################### Rotation Augmented Test Dataset ########################################
############################################################################################################################

    orientations = 12
    total_correct_class = torch.zeros(num_classes).to(device)
    total_samples_class = torch.zeros(num_classes).to(device)

    # Datasets
    datasets = get_datasets(data_path)
    test_loader = torch.utils.data.DataLoader(datasets['test'], batch_size = batch_size - batch_size%orientations)

    for batch_index, (inputs, labels) in enumerate(test_loader):

        t0 = time.time()
        n_correct_classwise, n_samples_classwise = test(inputs, labels)
        acc = torch.sum(n_correct_classwise) / (len(inputs) // orientations)
        total_correct_class += n_correct_classwise
        total_samples_class += n_samples_classwise
        t1 = time.time()

        logger.info(f'Test [{batch_index+1}/{len(test_loader)}] Time : {(t1-t0)*1e3:.1f} ms \t ACC={acc*100:.2f} %')
    total_accuracy = torch.sum(total_correct_class) / torch.sum(total_samples_class)
    logger.info(f'\n\nOverall Accuracy = {total_accuracy.item() * 100:.2f} %\n')


    total_accuracy_class = total_correct_class / total_samples_class
    for i in range(num_classes):
        logger.info(f'Accuracy of class {i} {total_accuracy_class[i] * 100:.2f} %')


        
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
    parser.add_argument("--interpolation", type=int, required=True)
    parser.add_argument("--restricted", type=int, required=True)

    args = parser.parse_args()
    sys.path.append(args.__dict__['model_path'])
    from model import Model, get_datasets

    main(**args.__dict__)
