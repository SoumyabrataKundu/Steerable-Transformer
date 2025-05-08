import os
import logging
import copy
import sys
import time
import h5py
import torch

from model import Model, get_datasets, Loss
from Steerable.Segmentation.metrics import Metrics

def main(data_path, batch_size, n_radius, max_m, learning_rate, weight_decay, num_epochs, num_workers, lr_decay_rate, lr_decay_schedule, save=0):
   
 ################################################################################################################################### 
 ##################################################### Logging #####################################################################
 ###################################################################################################################################
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
    
 ###################################################################################################################################
 ################################## Loading Model, Datasets, Loss and Optimizer ####################################################
 ###################################################################################################################################
    
    # Load the model
    model = Model(n_radius, max_m)
    num_classes = model.num_classes
    device = torch.device("cuda")
    model = model.to(device)
    
    logger.info("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    
    # Datasets
    datasets = get_datasets(data_path)

    ## Dataloaders
    train_loader = torch.utils.data.DataLoader(dataset = datasets['train'], batch_size = batch_size, shuffle = True, num_workers = num_workers)
    val_loader = None
    if datasets['val'] is not None:
         val_loader = torch.utils.data.DataLoader(dataset = datasets['val'], batch_size = batch_size, num_workers = num_workers)
    
    # Loss
    criterion = Loss 
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = 0, weight_decay = weight_decay)
    

    def get_learning_rate(epoch):
        return learning_rate * (lr_decay_rate ** (epoch // lr_decay_schedule))


####################################################################################################################################
######################################## Training and Evaluation Function ##########################################################
####################################################################################################################################
    
    def train_step(inputs, labels):
        model.train()
        
        # Pushing to GPU
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Forward Pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()

    def evaluate(dataloader):
        model.eval()
        metrics = Metrics(num_classes)
        total_loss = 0
        total_score = 0
        num_inputs = 0
        logger.info('\n Validation : \n')
        with torch.no_grad():

            for batch_idx, (inputs, labels) in enumerate(dataloader):
                # Pushing to GPU
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                t0 = time.time()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                t1 = time.time()
        
                total_loss += loss * len(inputs)
                preds = torch.argmax(outputs, dim=1)
                score = metrics.mDice(preds, labels)
                total_score += score * len(inputs)
                num_inputs += len(inputs)

                # Logging
                logger.info(f"Validation [{batch_idx+1}/{len(dataloader)}] Time : {(t1-t0)*1e3:.1f} ms Loss : {loss:.2f} Score : {score:.4f} <Score> : {total_score / num_inputs:.4f}")
            
        return total_loss / len(datasets['val']), total_score / len(datasets['val'])
    
####################################################################################################################################
################################################### Training Loop ##################################################################
####################################################################################################################################
    
    epoch = 0
    early_stop = 0
    best_val_loss, best_score = float('inf'), 0
    
    for epoch in range(epoch, num_epochs):
        
        lr = get_learning_rate(epoch)
        logger.info(f"learning rate = {lr}, weight decay = {weight_decay}, batch size = {train_loader.batch_size}")
        for p in optimizer.param_groups:
            p['lr'] = lr
        
        total_iteration_loss = 0
        total_iteration_time = 0
        
        for batch_index, (inputs, labels) in enumerate(train_loader):

            model.train()
            # Train Step
            t0 = time.time()
            iteration_loss = train_step(inputs, labels.long())
            t1 = time.time()

            total_iteration_time = total_iteration_time + (t1-t0)*1000
            avg_iteration_time = total_iteration_time / (batch_index + 1) 
            ## Loss
            total_iteration_loss += iteration_loss
            avg_loss = total_iteration_loss / (batch_index+1)

            # Logging
            logger.info(f"[{epoch+1}/{num_epochs}:{batch_index+1}/{len(train_loader)}] Time : {(t1-t0)*1e3:.1f} ms <Time> : {avg_iteration_time:.1f} ms\
                        LOSS={iteration_loss:.2f} <LOSS>={avg_loss:.2f}")
            torch.save(model.state_dict(), os.path.join(log_dir, "state.pkl"))
            
        # Evaluate
        if (epoch+1) % 3 == 0 or epoch == (num_epochs-1) or early_stop == 11:
            if val_loader is not None:
                ## Validation
                val_loss, score = evaluate(val_loader)
                if score>= best_score:
                    best_val_loss, best_score = val_loss, score
                    early_stop = 0
                    torch.save(model.state_dict(), os.path.join(log_dir, "best_state.pkl"))
                else:
                    early_stop += 1

                logger.info(f"\n\nLoss={val_loss:.2f} Best={best_val_loss:.2f} Score={score:.2f} Best={best_score:.2f}")                   
                print(f'epoch {epoch+1}/{num_epochs} avg loss : {avg_loss:.4f} \t val loss : {val_loss:.4f} score : {score:.4f} \t best loss : {best_val_loss:.4f} best score : {best_score:.4f} {"*" if score==best_score else ""}')

                if early_stop == 11:
                   print(f"\n\nStopped at epoch {epoch+1}.\n")
                   break

            else :
                print(f'epoch {epoch+1}/{num_epochs} avg loss : {avg_loss:.4f}')
                
                logger.info("\n\n")
            
        logger.info("\n\n")

    ########################################################################################################################
    ############################################### Test Function ##########################################################
    ########################################################################################################################

    def test(inputs, labels):
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels).item()
            probs = torch.softmax(outputs, dim=1)

        return probs, loss

    ###########################################################################################################################
    ################################################### Testing Loop ##########################################################
    ###########################################################################################################################
    
    total_loss = 0
    model.eval()
    metrics = Metrics(num_classes)
    total_score = torch.zeros(num_classes)

    if datasets['val'] is not None:
        model.load_state_dict(torch.load(os.path.join('log', 'best_state.pkl')))
    else:
        model.load_state_dict(torch.load(os.path.join('log', 'state.pkl')))
    test_loader = torch.utils.data.DataLoader(datasets['test'], batch_size = batch_size, num_workers = num_workers)
    
    logger.info(f"\n\n\nTesting:\n")

    save = bool(int(save))
    if save:
        run, config = os.path.split(os.getcwd())
        run = os.path.basename(run)
        f = h5py.File(os.path.join(log_dir, 'probs.hdf5'), 'w')
        prob_dataset = None
 
    for batch_index, (inputs, labels) in enumerate(test_loader):

        inputs = inputs.to(device)
        labels = labels.to(device)

        t0 = time.time()
        probs, current_loss = test(inputs, labels)
        preds = torch.argmax(probs, dim=1)
        t1 = time.time()

        if save:
            if prob_dataset is None:
                 f.create_dataset('probs', (0, ) + probs.shape[1:], maxshape=(None,) +  probs.shape[1:], chunks=True)
                 prob_dataset = f['probs']

            prob_dataset.resize((len(prob_dataset) + len(probs),) + probs_dataset.shape[1:])
            prob_dataset[-len(outputs):] = probs.cpu()

        metrics.add_to_confusion_matrix(preds, labels)
        score = metrics.dice_per_class(preds, labels)
        total_score += score * len(inputs)
        total_loss += current_loss * len(inputs)

        logger.info(f'Test [{batch_index+1}/{len(test_loader)}] Time : {(t1-t0)*1e3:.1f} ms  Loss={current_loss:.2f} \t Score : {score} \t <Score> : {torch.mean(score[1:]).item():.4f}')

    if save:   
        f.close() 
    avg_loss = total_loss / len(datasets['test'])
    avg_score_per_class = total_score_per_class / len(datasets['test'])
    avg_score = 

    print(f'\n\nTesting Loss = {avg_loss:.4f}')
    print(f"\nScore per class = {avg_score}")
    print(f"Avg Score = {torch.mean(avg_dice[1:]).item():.4f}")
    print(f"\nGlobal Score per class = {metrics.micro_per_class()}")
    print(f"Global Score = {metrics.micro():.4f}")

############################################################################################################################
################################################### Argument Parser ########################################################
############################################################################################################################

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--n_radius", type=int,required=True)
    parser.add_argument("--max_m", type=int, required=True)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--lr_decay_rate", type=float, default=0.5)
    parser.add_argument("--lr_decay_schedule", type=int, default=20)
    parser.add_argument("--save", type=int, default=0)

    args = parser.parse_args()
    main(**args.__dict__)
