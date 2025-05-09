import os
import logging
import copy
import time
import h5py
import torch

from model import Model, get_datasets, Loss
import Steerable.utils.Metrics as Metrics

def main(data_path, batch_size, n_radius, max_m, num_workers, metric_type, save=0):
    #####################################################################################################################################
    ############################################################## Logging ##############################################################
    #####################################################################################################################################
    arguments = copy.deepcopy(locals())
    
    log_dir = os.path.join('log/')
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    
    # Creating the logger
    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    ch = logging.StreamHandler()
    logger.addHandler(ch)
    fh = logging.FileHandler(os.path.join(log_dir, "log.txt"), mode = "w")
    logger.addHandler(fh)

    logger.info("%s", repr(arguments))
    logger.info("\n\n")
    
    #####################################################################################################################################
    ################################################## Loading Model, Dataset and Loss ##################################################
    #####################################################################################################################################
    
    # Load the model
    model = Model(n_radius, max_m)
    num_classes = model.num_classes
    device = torch.device("cuda")
    model = model.to(device)
    
    # DataLoader
    datasets = get_datasets(data_path)
    test_loader = torch.utils.data.DataLoader(datasets['test'], batch_size = batch_size, num_workers = num_workers)
    
    # Loading Best Model
    if datasets['val'] is not None:
        model.load_state_dict(torch.load(os.path.join('log', 'best_state.pkl')))
    else:
        model.load_state_dict(torch.load(os.path.join('log', 'state.pkl')))
    logger.info("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
         
    # Loss
    criterion = Loss

    #####################################################################################################################################
    ########################################################### Testing Loop ############################################################
    #####################################################################################################################################
    
    def test_step(inputs, labels):
        model.eval()
        
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels).item()
            probs = torch.softmax(outputs, dim=1)

        return probs, loss
    
    
    # Metrics
    metrics = Metrics(num_classes, metric_type)
    total_loss, total_score, total_score_per_class, num_inputs = 0, 0, torch.zeros(num_classes), 0

    # Saving
    save = bool(int(save))
    if save:
        f = h5py.File(os.path.join(log_dir, 'probs.hdf5'), 'w')
        prob_dataset = None
 
    # Testing
    logger.info(f"\n\n\nTesting:\n")
    for batch_index, (inputs, labels) in enumerate(test_loader):
        
        t0 = time.time()
        probs, current_loss = test_step(inputs, labels)
        t1 = time.time()
        
        preds = torch.argmax(probs, dim=1).detach().cpu()
        metrics.add_to_confusion_matrix(preds, labels)
        score_per_class = metrics.macro_per_class(preds, labels)
        score = metrics.macro(preds, labels)
        
        total_score_per_class += score_per_class * len(inputs)
        total_score += score * len(inputs)
        total_loss += current_loss * len(inputs)
        num_inputs += len(inputs)

        if save:
            if prob_dataset is None:
                 f.create_dataset('probs', (0, ) + probs.shape[1:], maxshape=(None,) +  probs.shape[1:], chunks=True)
                 prob_dataset = f['probs']

            prob_dataset.resize((len(prob_dataset) + len(probs),) + probs_dataset.shape[1:])
            prob_dataset[-len(probs):] = probs.cpu()

        logger.info(f'Test [{batch_index+1}/{len(test_loader)}] '
                    f'Time : {(t1-t0)*1e3:.1f} ms  Loss={current_loss:.2f}\t'
                    f'Score : {score_per_class} <Score> : {total_score / num_inputs:.4f}')

    if save:   
        f.close()
        
    avg_loss = total_loss / len(datasets['test'])
    avg_score_per_class = total_score_per_class / len(datasets['test'])
    avg_score = total_score / len(datasets['test'])

    logger.info(f'\n\nTesting Loss = {avg_loss:.4f}')
    logger.info(f"\nScore per class = {avg_score_per_class}")
    logger.info(f"Avg Score = {avg_score:.4f}")
    logger.info(f"\nGlobal Score per class = {metrics.micro_per_class()}")
    logger.info(f"Global Score = {metrics.micro():.4f}")

    print(f'\n\nTesting Loss = {avg_loss:.4f}')
    print(f"\nScore per class = {avg_score_per_class}")
    print(f"Avg Score = {avg_score:.4f}")
    print(f"\nGlobal Score per class = {metrics.micro_per_class()}")
    print(f"Global Score = {metrics.micro():.4f}")

#####################################################################################################################################
########################################################## Argument Parser ##########################################################
#####################################################################################################################################
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--n_radius", type=int,required=True)
    parser.add_argument("--max_m", type=int, required=True)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--metric_type", type=str, default='dice')
    parser.add_argument("--save", type=int, default=0)

    args = parser.parse_args()
    main(**args.__dict__)
