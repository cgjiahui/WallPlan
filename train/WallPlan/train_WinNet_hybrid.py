import torch
from torch.utils.data import DataLoader
from inspect import getsource
from torchnet import meter
from tqdm import tqdm
import numpy as np
import models
import utils
import time
import csv
import os
from config_WinNet import WindowConfig
from data import WindowApplicationDataset
win_opt=WindowConfig()
log=utils.log


def train(**kwargs):
    "Train with other constraints"
    name = time.strftime('junction_train_%Y_%m_%d_%H_%M_%S')
    log_file = open(f"{win_opt.save_log_root}/{name}.txt", 'w')
    win_opt.parse(kwargs, log_file)
    start_time = time.strftime("%b %d %Y %H:%M:%S")
    log(log_file, f'Training start time: {start_time}')

    # step1: configure model
    log(log_file, 'Building model...')

    model=models.model(
        module_name=win_opt.module_name,
        model_name=win_opt.model_name,
        num_classes=win_opt.num_classes,
        num_channels=win_opt.num_channels
    )

    if win_opt.multi_GPU:
        model_parallel = models.ParallelModule(model=model)
        if win_opt.load_model_path:
            model_parallel.load_model(win_opt.load_model_path)
        model = model_parallel.model
    else:
        if win_opt.load_model_path:
            model.load_model(win_opt.load_model_path)

    model.cuda()

    # step2: data
    log(log_file, 'Building dataset...')
    win_train_data = WindowApplicationDataset(data_root=win_opt.train_data_root, mask_size=win_opt.mask_size,mode=win_opt.mode,constraint_split=win_opt.constraint_split)
    win_val_data = WindowApplicationDataset(data_root=win_opt.val_data_root, mask_size=win_opt.mask_size,mode=win_opt.mode,constraint_split=win_opt.constraint_split)

    log(log_file, 'Building data loader...')
    train_dataloader = DataLoader(
        win_train_data,
        win_opt.batch_size,
        shuffle=True,
        num_workers=win_opt.num_workers
    )

    val_dataloader = DataLoader(
        win_val_data,
        win_opt.batch_size,
        shuffle=True,
        num_workers=win_opt.num_workers
    )

    # step3: criterion and optimizer
    log(log_file, 'Building criterion and optimizer...')
    lr = win_opt.lr_base
    optimizer = torch.optim.Adam(
        list(model.parameters()) ,
        lr=lr,
        weight_decay=win_opt.weight_decay
    )
    current_epoch = win_opt.current_epoch
    weight = torch.ones(2)
    weight[1] = 2.5


    weight = weight.cuda()
    criterion = torch.nn.CrossEntropyLoss(weight=weight)
    loss_meter = meter.AverageValueMeter()

    # step4: training
    log(log_file, 'Starting to train...')
    if current_epoch == 0 and os.path.exists(win_opt.result_file):
        os.remove(win_opt.result_file)
    result_file = open(win_opt.result_file, 'a', newline='')
    writer = csv.writer(result_file)
    if current_epoch == 0:
        data_name = ['Epoch', 'Average Loss', 'Predict Accuracy', 'Number of Predict Categoey Right', \
                     'Number of Target Categoey', 'Number of Predict Categoey', 'Category Accuracy',
                     'Category Proportion']
        writer.writerow(data_name)
        result_file.flush()

    garmmar = 0.9
    max_sum=0
    while current_epoch < win_opt.max_epoch:
        train_dataloader_change=train_dataloader
        current_epoch += 1
        running_loss = 0.0
        loss_meter.reset()
        log(log_file)
        log(log_file, f'Training epoch: {current_epoch}')
        for i, (input, target) in tqdm(enumerate(train_dataloader_change)):
            input = input.cuda()
            target = target.cuda()
            optimizer.zero_grad()
            score_model = model(input)
            loss = criterion(score_model, target.long())
            loss.backward()
            optimizer.step()

            # log info
            running_loss += loss.item()
            if i % win_opt.print_freq == win_opt.print_freq - 1:
                log(log_file, f'loss {running_loss / win_opt.print_freq:.5f}')
                running_loss = 0.0
            loss_meter.add(loss.item())
        if current_epoch % win_opt.save_freq == 0:
            if win_opt.multi_GPU:
                model_parallel.save_model(current_epoch)
            else:
                model.save_model(current_epoch)
        average_loss = round(loss_meter.value()[0], 5)
        log(log_file, f'Average Loss: {average_loss}')

        # validate
        if current_epoch % win_opt.val_freq == 0:
            predict_accuracy, num_predict_categoey_right, num_target_categoey, num_predict_categoey, \
            categoey_accuracy, categoey_proportion = val(model, val_dataloader, log_file)
            results = [current_epoch, average_loss, predict_accuracy, num_predict_categoey_right, \
                       num_target_categoey, num_predict_categoey, categoey_accuracy, categoey_proportion]
            writer.writerow(results)
            result_file.flush()
        if max_sum<categoey_accuracy+categoey_proportion and current_epoch%10!=0:
            max_sum=categoey_accuracy+categoey_proportion
            model.save_model(current_epoch)
        # update learning rate
        if win_opt.update_lr:
            if current_epoch % win_opt.lr_decay_freq == 0:
                lr = lr * (1 - float(current_epoch) / win_opt.max_epoch) ** 1.5
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                    log(log_file, f'Updating learning rate: {lr}')

    end_time = time.strftime("%b %d %Y %H:%M:%S")
    log(log_file, f'Training end time: {end_time}')
    log_file.close()
    result_file.close()

def val(model, dataloader, file):
    model.eval()
    predict_accuracy = 0
    num_predict_categoey_right = 0
    num_target_categoey = 0
    num_predict_categoey = 0
    softmax = torch.nn.Softmax(dim=1)
    for _, (input, target) in enumerate(dataloader):
        batch_size = input.shape[0]
        with torch.no_grad():
            input = input.cuda()
            score_model = model(input)
        score_softmax = softmax(score_model)
        output = score_softmax.cpu().numpy()
        predict = np.argmax(output, axis=1)
        target = target.numpy()
        for i in range(batch_size):
            num_predict = np.sum(predict[i] == target[i])
            predict_accuracy += (num_predict / (input.shape[2] * input.shape[3]))
            for k in range(1, 2):
                num_predict_categoey_right += np.sum((predict[i] == k) & (target[i] == k))
                num_target_categoey += np.sum(target[i] == k)
                num_predict_categoey += np.sum(predict[i] == k)
    model.train()
    predict_accuracy = round(predict_accuracy / len(dataloader.dataset), 5)
    num_predict_categoey_right = int(num_predict_categoey_right / len(dataloader.dataset))
    num_target_categoey = int(num_target_categoey / len(dataloader.dataset))
    num_predict_categoey = int(num_predict_categoey / len(dataloader.dataset))
    if num_target_categoey != 0:
        category_accuracy = round(num_predict_categoey_right / num_target_categoey, 5)
    else:
        category_accuracy = 0
    if num_predict_categoey != 0:
        category_proportion = round(num_predict_categoey_right / num_predict_categoey, 5)
    else:
        category_proportion = 0
    log(file, f'Predict Accuracy: {predict_accuracy}')
    log(file, f'Number of Predict Categoey Right: {num_predict_categoey_right}')
    log(file, f'Number of Target Categoey: {num_target_categoey}')
    log(file, f'Number of Predict Categoey: {num_predict_categoey}')
    log(file, f'Category Accuracy: {category_accuracy}')
    log(file, f'Category Proportion: {category_proportion}')
    return predict_accuracy, num_predict_categoey_right, num_target_categoey, num_predict_categoey, \
           category_accuracy, category_proportion
def help():
    print("""
    usage : python file.py <function> [--args=value]
    <function> := train | help
    example: 
            python {0} train --lr=0.01
            python {0} help
    avaiable args:""".format(__file__))
    source = (getsource(win_opt.__class__))
    print(source)
if __name__ == '__main__':
    train()