import torch
from torch.utils.data import DataLoader
from inspect import getsource
from torchnet import meter
from tqdm import tqdm
import numpy as np

import models
import utils
import fire
import time
import csv
import os
from config_LabelNet import LabelNetConfig
from data import LabelNetDataset

LabelNet_opt = LabelNetConfig()
log = utils.log

def train(**kwargs):
    name = time.strftime('junction_train_%Y_%m_%d_%H_%M_%S')
    log_file = open(f"{LabelNet_opt.save_log_root}/{name}.txt", 'w')
    LabelNet_opt.parse(kwargs, log_file)
    start_time = time.strftime("%b %d %Y %H:%M:%S")
    log(log_file, f'Training start time: {start_time}')

    log(log_file, 'Building model...')
    model=models.model(
        module_name=LabelNet_opt.module_name,
        model_name=LabelNet_opt.model_name,
        num_classes=LabelNet_opt.num_classes,
        num_channels=LabelNet_opt.num_channels
    )

    if LabelNet_opt.multi_GPU:
        model_parallel = models.ParallelModule(model=model)
        if LabelNet_opt.load_model_path:
            model_parallel.load_model(LabelNet_opt.load_model_path)
        model = model_parallel.model
    else:
        if LabelNet_opt.load_model_path:
            model.load_model(LabelNet_opt.load_model_path)
    model.cuda()

    log(log_file, 'Building dataset...')
    seman_train_data = LabelNetDataset(data_root=LabelNet_opt.train_data_root, mask_size=LabelNet_opt.mask_size)
    seman_val_data = LabelNetDataset(data_root=LabelNet_opt.val_data_root, mask_size=LabelNet_opt.mask_size)

    log(log_file, 'Building data loader...')
    train_dataloader = DataLoader(
        seman_train_data,
        LabelNet_opt.batch_size,
        shuffle=True,
        num_workers=LabelNet_opt.num_workers
    )

    val_dataloader = DataLoader(
        seman_val_data,
        LabelNet_opt.batch_size,
        shuffle=True,
        num_workers=LabelNet_opt.num_workers
    )

    log(log_file, 'Building criterion and optimizer...')
    lr = LabelNet_opt.lr_base
    optimizer = torch.optim.Adam(
        list(model.parameters()),
        lr=lr,
        weight_decay=LabelNet_opt.weight_decay
    )
    current_epoch = LabelNet_opt.current_epoch
    weight = torch.ones(7 + 1)
    weight[1]=1.3
    weight[7]=10

    weight=weight.cuda()
    criterion = torch.nn.CrossEntropyLoss(weight=weight)
    loss_meter = meter.AverageValueMeter()

    log(log_file, 'Starting to train...')
    if current_epoch == 0 and os.path.exists(LabelNet_opt.result_file):
        os.remove(LabelNet_opt.result_file)
    result_file = open(LabelNet_opt.result_file, 'a', newline='')
    writer = csv.writer(result_file)
    if current_epoch == 0:
        data_name = ['Epoch', 'Average Loss', 'Predict Accuracy', 'Number of Predict Categoey Right', \
                     'Number of Target Categoey', 'Number of Predict Categoey', 'Category Accuracy',
                     'Category Proportion']
        writer.writerow(data_name)
        result_file.flush()
    max_sum = 0
    while current_epoch < LabelNet_opt.max_epoch:
        current_epoch += 1
        running_loss = 0.0
        loss_meter.reset()
        log(log_file)
        log(log_file, f'Training epoch: {current_epoch}')

        for i, (input, target) in tqdm(enumerate(train_dataloader)):
            input = input.cuda()
            target = target.cuda()
            optimizer.zero_grad()
            score_model = model(input)
            loss = criterion(score_model, target.long())
            loss.backward()
            optimizer.step()

            # log info
            running_loss += loss.item()
            if i % LabelNet_opt.print_freq == LabelNet_opt.print_freq - 1:
                log(log_file, f'loss {running_loss / LabelNet_opt.print_freq:.5f}')
                running_loss = 0.0
            loss_meter.add(loss.item())

        if current_epoch % LabelNet_opt.save_freq == 0:
            if LabelNet_opt.multi_GPU:
                model_parallel.save_model(current_epoch)
            else:
                model.save_model(current_epoch)
        average_loss = round(loss_meter.value()[0], 5)
        log(log_file, f'Average Loss: {average_loss}')

        # validate
        if current_epoch % LabelNet_opt.val_freq == 0:
            predict_accuracy, num_predict_categoey_right, num_target_categoey, num_predict_categoey, \
            categoey_accuracy, categoey_proportion = val(model, val_dataloader, log_file)
            results = [current_epoch, average_loss, predict_accuracy, num_predict_categoey_right, \
                       num_target_categoey, num_predict_categoey, categoey_accuracy, categoey_proportion]
            writer.writerow(results)
            result_file.flush()

        if max_sum < categoey_accuracy + categoey_proportion and current_epoch % 10 != 0:
            max_sum = categoey_accuracy + categoey_proportion
            model.save_model(current_epoch)

        # update learning rate
        if LabelNet_opt.update_lr:
            if current_epoch % LabelNet_opt.lr_decay_freq == 0:
                lr = lr * (1 - float(current_epoch) / LabelNet_opt.max_epoch) ** 1.5
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
            for k in range(1,6+1):
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
        category_accuracy = "nan"
    if num_predict_categoey != 0:
        category_proportion = round(num_predict_categoey_right / num_predict_categoey, 5)
    else:
        category_proportion = "nan"
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

    source = (getsource(LabelNet_opt.__class__))
    print(source)
if __name__ == '__main__':
    train()