# -*- coding: utf-8 -*-
class DefaultConfig(object):
    net_work = 'RETFound'
    num_classes = 5
    num_epochs = 10
    batch_size = 8
    validation_step = 1


    root = "D:/"
    root_test = "D:/"    
    train_file = "Datasets/DC_pred_train.csv"
    val_file = "Datasets/DC_pred_val.csv"
    test_file = "Datasets/DC_pred_test.csv"
    seed = "1234"


    lr = 1e-4
    lr_mode = 'poly'
    momentum = 0.9
    weight_decay = 1e-4


    save_model_path = 'ModelSaved'
    log_dirs = 'Log'

    pretrained = False
    pretrained_model_path = None

    cuda = 0
    num_workers = 4
    use_gpu = True

    trained_model_path = ''
    predict_fold = 'predict_mask'
