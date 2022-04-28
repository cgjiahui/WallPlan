import warnings
import utils
class LabelNetConfig(object):
    train_data_root = '../pickle/train_dataset'
    val_data_root = '../pickle/val_dataset'
    train_data_root='G:/sjh/origin_dataset/dataset_8w_11/Junction_graph_8w_split/train'
    val_data_root='G:/sjh/origin_dataset/dataset_8w_11/Junction_graph_8w_split/val'
    save_log_root = 'log'
    result_file = 'result_junction.csv'
    module_name = 'LabelNet'
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

    # "Normal"
    num_classes=8
    num_channels=6

    # "Application"
    # num_classes=8
    # num_channels = 10

    def parse(self, kwargs, file):
        "update parameters"
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                utils.log(file, f'{k}: {getattr(self, k)}')
