import logging
import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset
from pathlib import Path

import utils
from configs import Config
from tgcn_model import GCN_muti_att
# from tgcn_dense import TGCN_GRU
from tgcn_model_gru import TGCN_GRU
from lstm import SingleLSTM
from sign_dataset import Sign_Dataset
from sign_dataset_mp import SignDatasetMP
from train_utils import train_stacked_gru, validation_stacked_gru

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def run(split_file, pose_data_root, configs, save_model_to=None):
    checkpoint_path = Path('checkpoints') / subset

    epochs = configs.max_epochs
    log_interval = configs.log_interval
    num_samples = configs.num_samples
    hidden_size = configs.hidden_size
    drop_p = configs.drop_p
    num_stages = configs.num_stages

    # setup dataset
    print("LOADING DATASETS...")

    # WLASL DATASETS
    train_dataset = Sign_Dataset(index_file_path=split_file, split='train', pose_root=pose_data_root,
                                 img_transforms=None, video_transforms=None, num_samples=num_samples)

    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                                    shuffle=True)

    val_dataset = Sign_Dataset(index_file_path=split_file, split='val', pose_root=pose_data_root,
                               img_transforms=None, video_transforms=None,
                               num_samples=num_samples,
                               sample_strategy='k_copies')
    val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=configs.batch_size,
                                                  shuffle=True)

    logging.info('\n'.join(['Class labels are: '] + [(str(i) + ' - ' + label) for i, label in
                                                     enumerate(train_dataset.label_encoder.classes_)]))



    # setup the model
    hidden_size = 256
    model = SingleLSTM(input_size=110, hidden_size=hidden_size, num_classes=100, p_dropout=0.2).cuda()
    print("MODEL LOADED...")

    # setup training parameters, learning rate, optimizer, scheduler
    lr = configs.init_lr
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=configs.adam_eps, weight_decay=configs.adam_weight_decay)

    # record training process
    epoch_train_losses = []
    epoch_train_scores = []
    epoch_val_losses = []
    epoch_val_scores = []

    best_test_acc = 0
    # start training
    for epoch in range(int(epochs)):
        # train, test model

        print('STARTING TRAINING...')
        train_losses, train_scores, train_gts, train_preds = train_stacked_gru(log_interval, model,
                                                                   train_data_loader, optimizer, epoch)
        print('STARTING TESTING...')
        val_loss, val_score, val_gts, val_preds, incorrect_samples = validation_stacked_gru(model,
                                                                                val_data_loader, epoch,
                                                                                save_to=save_model_to)

        logging.info('========================\nEpoch: {} Average loss: {:.4f}'.format(epoch, val_loss))
        logging.info('Top-1 acc: {:.4f}'.format(100 * val_score[0]))
        logging.info('Top-3 acc: {:.4f}'.format(100 * val_score[1]))
        logging.info('Top-5 acc: {:.4f}'.format(100 * val_score[2]))
        logging.info('Top-10 acc: {:.4f}'.format(100 * val_score[3]))
        logging.info('Top-30 acc: {:.4f}'.format(100 * val_score[4]))
        logging.debug('mislabelled val. instances: ' + str(incorrect_samples))

        # save results
        epoch_train_losses.append(train_losses)
        epoch_train_scores.append(train_scores)
        epoch_val_losses.append(val_loss)
        epoch_val_scores.append(val_score[0])

        # save all train test results
        model_name = 'lstm'
        np.save(f'output/{model_name}_epoch_training_losses.npy', np.array(epoch_train_losses))
        np.save(f'output/{model_name}_epoch_training_scores.npy', np.array(epoch_train_scores))
        np.save(f'output/{model_name}_epoch_test_loss.npy', np.array(epoch_val_losses))
        np.save(f'output/{model_name}_epoch_test_score.npy', np.array(epoch_val_scores))

        if val_score[0] > best_test_acc:
            best_test_acc = val_score[0]
            best_epoch_num = epoch

            if val_score[0] > 0.3:
                torch.save(model.state_dict(), checkpoint_path / f'{model_name}-epoch={best_epoch_num}-val_acc={best_test_acc}-hidden={hidden_size}.pth')

    utils.plot_curves(model=model_name)

    class_names = train_dataset.label_encoder.classes_
    utils.plot_confusion_matrix(train_gts, train_preds, classes=class_names, normalize=False,
                                save_to=f'output/{model_name}_train-conf-mat')
    utils.plot_confusion_matrix(val_gts, val_preds, classes=class_names, normalize=False, save_to=f'output/{model_name}_val-conf-mat')


if __name__ == "__main__":
    subset = 'asl100'

    split_file = os.path.join('../../data/splits/{}.json'.format(subset))
    pose_data_root = os.path.join('../../data/pose_per_individual_videos')
    config_file = os.path.join('configs/{}.ini'.format(subset))
    configs = Config(config_file)

    logging.basicConfig(filename='output/{}.log'.format(os.path.basename(config_file)[:-4]), level=logging.DEBUG, filemode='w+')

    logging.info('Calling main.run()')
    run(split_file=split_file, configs=configs, pose_data_root=pose_data_root)
    logging.info('Finished main.run()')
