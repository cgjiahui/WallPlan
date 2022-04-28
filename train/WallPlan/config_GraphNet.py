import warnings
import utils

class GraphNetConfig(object):
    train_data_root='G:/sjh/origin_dataset/dataset_8w_11/Junction_graph_8w_split/train'
    val_data_root='G:/sjh/origin_dataset/dataset_8w_11/Junction_graph_8w_split/val'
    save_log_root = 'log'
    result_file = 'result_junction.csv'
    module_name = 'GraphNet'
    model_name = 'dlink34no'
    load_model_path = None
    load_connect_path = None
    mask_size = 9

    multi_GPU = False
    batch_size = 16
    num_workers = 3
    print_freq = 300

    max_epoch = 60
    current_epoch = 0
    save_freq = 10
    val_freq = 1

    update_lr = True
    lr_decay_freq = 1
    lr_base = 1.2e-4
    weight_decay = 1e-4

    process_splits = [0.4, 0.5]
    constraint_split = [0.1, 0.3, 0.5]

    "Normal"
    num_classes=3
    num_channels=8

    # "Application"
    # num_classes = 3
    # num_channels = 12

    Label_model_pth="../../trained_model/OnlyBoundary/LabelNet.pth"
    Label_model_4application_pth=""

    def parse(self, kwargs, file):
        """
        根据字典kwargs 更新 config参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                utils.log(file, f'{k}: {getattr(self, k)}')
