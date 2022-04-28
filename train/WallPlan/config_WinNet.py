import warnings
import utils

class WindowConfig(object):
    train_data_root='G:/sjh/origin_dataset/dataset_8w_11/Junction_graph_8w_split/train'
    val_data_root='G:/sjh/origin_dataset/dataset_8w_11/Junction_graph_8w_split/val'
    save_log_root = 'log'
    result_file = 'result_junction.csv'
    module_name = 'WinNet'
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

    constraint_split=[0.1,0.3,0.5]

    "boundary constraints"
    #mode 0 for Living window training, 1 for other window training
    mode = 1
    num_classes = 2
    num_channels = (4+1) if mode==1 else 4

    "mixed constraints"
    # mode = 0
    # num_classes = 2
    # num_channels = (8+1) if mode==1 else 8

    update_lr = True
    lr_decay_freq = 1
    lr_base = 1.2e-4
    weight_decay = 1e-4

    def parse(self, kwargs, file):
        """
        update parameters
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                utils.log(file, f'{k}: {getattr(self, k)}')