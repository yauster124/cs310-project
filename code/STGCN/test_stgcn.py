import os
from pathlib import Path

from configs import Config
from sign_dataset import Sign_Dataset
import numpy as np
import torch
from sklearn.metrics import accuracy_score

from stgcn_model import Model
from stacked_gru import StackedGRU
from tgcn_model import GCN_muti_att

import glob


def test(model, test_loader):
    # set model as testing mode
    model.eval()

    all_y = []
    all_y_pred = []
    all_video_ids = []
    all_pool_out = []

    num_copies = 4

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            print('starting batch: {}'.format(batch_idx))
            # distribute data to device
            X, y, video_ids = data
            X, y = X.cuda(), y.cuda().view(-1, )

            all_output = []

            stride = X.size()[2] // num_copies

            for i in range(num_copies):
                X_slice = X[:, :, i * stride: (i + 1) * stride]
                X_slice = torch.stack(torch.chunk(X_slice, 50, 2))
                X_slice = torch.permute(X_slice, (1, 0, 2, 3))
                X_slice = torch.flatten(X_slice, start_dim=2)
                output = model(X_slice)
                all_output.append(output)

            all_output = torch.stack(all_output, dim=1)
            output = torch.mean(all_output, dim=1)

            y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(y_pred)
            all_video_ids.extend(video_ids)
            all_pool_out.extend(output)

    # compute accuracy
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0).squeeze()
    all_pool_out = torch.stack(all_pool_out, dim=0).cpu().data.numpy()

    all_y = all_y.cpu().data.numpy()
    all_y_pred = all_y_pred.cpu().data.numpy()

    # top-k accuracy
    top1acc = accuracy_score(all_y, all_y_pred)
    top3acc = compute_top_n_accuracy(all_y, all_pool_out, 3)
    top5acc = compute_top_n_accuracy(all_y, all_pool_out, 5)
    top10acc = compute_top_n_accuracy(all_y, all_pool_out, 10)

    # show information
    print('\nVal. set ({:d} samples): top-1 Accuracy: {:.2f}%\n'.format(len(all_y), 100 * top1acc))
    print('\nVal. set ({:d} samples): top-3 Accuracy: {:.2f}%\n'.format(len(all_y), 100 * top3acc))
    print('\nVal. set ({:d} samples): top-5 Accuracy: {:.2f}%\n'.format(len(all_y), 100 * top5acc))
    print('\nVal. set ({:d} samples): top-10 Accuracy: {:.2f}%\n'.format(len(all_y), 100 * top10acc))

def test_multi(model1, model2, model3, test_loader, w):
    # set model as testing mode
    model1.eval()
    model2.eval()

    all_y = []
    all_y_pred = []
    all_video_ids = []
    all_pool_out = []

    num_copies = 4

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            print('starting batch: {}'.format(batch_idx))
            # distribute data to device
            # X1, X2, y, video_ids = data
            # X1, X2, y = X1.cuda(), X2.cuda(), y.cuda().view(-1, )
            X1, y, video_ids = data
            X1, y = X1.cuda(),  y.cuda().view(-1, )

            all_output = []

            stride1 = X1.size()[2] // num_copies
            # stride2 = X2.size()[2] // num_copies

            for i in range(num_copies):
                X1_slice = X1[:, :, i * stride1: (i + 1) * stride1]
                X1_slice_TGCN = X1_slice
                X1_slice = torch.stack(torch.chunk(X1_slice, 50, 2))
                X1_slice = torch.permute(X1_slice, (1, 0, 2, 3))
                X1_slice = torch.flatten(X1_slice, start_dim=2)

                # X2_slice = X2[:, :, i * stride2: (i + 1) * stride2]
                # X2_slice = torch.stack(torch.chunk(X2_slice, 50, 2))
                # X2_slice = torch.permute(X2_slice, (1, 0, 2, 3))
                # X2_slice = torch.flatten(X2_slice, start_dim=2)

                output1 = model1(X1_slice)
                # output2 = model2(X2_slice)
                output2 = model2(X1_slice)
                all_output.append(output1)
                all_output.append(w*output2)

            all_output = torch.stack(all_output, dim=1)
            output = torch.mean(all_output, dim=1)

            y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(y_pred)
            all_video_ids.extend(video_ids)
            all_pool_out.extend(output)

    # compute accuracy
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0).squeeze()
    all_pool_out = torch.stack(all_pool_out, dim=0).cpu().data.numpy()

    all_y = all_y.cpu().data.numpy()
    all_y_pred = all_y_pred.cpu().data.numpy()

    # top-k accuracy
    top1acc = accuracy_score(all_y, all_y_pred)
    top3acc = compute_top_n_accuracy(all_y, all_pool_out, 3)
    top5acc = compute_top_n_accuracy(all_y, all_pool_out, 5)
    top10acc = compute_top_n_accuracy(all_y, all_pool_out, 10)

    # show information
    print('\nVal. set ({:d} samples): top-1 Accuracy: {:.2f}%\n'.format(len(all_y), 100 * top1acc))
    print('\nVal. set ({:d} samples): top-3 Accuracy: {:.2f}%\n'.format(len(all_y), 100 * top3acc))
    print('\nVal. set ({:d} samples): top-5 Accuracy: {:.2f}%\n'.format(len(all_y), 100 * top5acc))
    print('\nVal. set ({:d} samples): top-10 Accuracy: {:.2f}%\n'.format(len(all_y), 100 * top10acc))

def compute_top_n_accuracy(truths, preds, n):
    best_n = np.argsort(preds, axis=1)[:, -n:]
    ts = truths
    successes = 0
    for i in range(ts.shape[0]):
        if ts[i] in best_n[i, :]:
            successes += 1
    return float(successes) / ts.shape[0]


if __name__ == '__main__':

    # ASSUME CWD IS cs310/code/WLASL/code/TGCN
    root = Path('../../../../../../large/u2008310')
    code_root = Path('../..')
    trained_on = 'asl300'

    # checkpoints = [
    #     'checkpoints/asl100/multi_modal/stgcn-epoch=142-val_acc=0.5355029585798816-hidden=1024.pth',
    #     'checkpoints/asl100/multi_modal/stacked_gru-epoch=131-val_acc=0.5266272189349113-hidden=1024.pth'
    # ]
    checkpoints = [
        'checkpoints/asl100/multi_modal/stgcn-epoch=142-val_acc=0.5355029585798816-hidden=1024.pth',
        'checkpoints/asl100/multi_modal/stacked_gru-epoch=112-val_acc=0.6005917159763313-hidden=1024.pth'
    ]
    split_file = root / f'data/splits/{trained_on}.json'
    pose_data_root = root / 'data/pose_per_individual_videos'
    config_file = f'configs/{trained_on}.ini'

    print(os.path.exists(config_file))
    print(os.getcwd())

    configs = Config(config_file)

    num_samples = configs.num_samples
    hidden_size = configs.hidden_size
    drop_p = configs.drop_p
    num_stages = configs.num_stages
    batch_size = configs.batch_size

    dataset = Sign_Dataset(index_file_path=split_file, split='test', pose_root=pose_data_root,
                           img_transforms=None, video_transforms=None,
                           num_samples=num_samples,
                           sample_strategy='k_copies',
                           test_index_file=split_file)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    # setup the model
    graph_args = {'layout': 'openpose', 'strategy': 'spatial'}
    model = Model(in_channels=3,
                  num_class=300,
                  graph_args=graph_args,
                  edge_importance_weighting=True).cuda()
    model2 = StackedGRU(input_size=159, 
                        hidden_size=1024, 
                        num_classes=300, 
                        p_dropout=0.2).cuda()
    model3 = GCN_muti_att(input_feature=num_samples*3, hidden_feature=num_samples*3,
                         num_class=100, p_dropout=drop_p, num_stage=num_stages).cuda()

    # MULTI MODAL
    # print('LOADING WEIGHTS...')
    # checkpoint1 = torch.load(checkpoints[0])
    # checkpoint2 = torch.load(checkpoints[1])
    # model.load_state_dict(checkpoint1)
    # model2.load_state_dict(checkpoint2)
    # print('WEIGHTS LOADED')
    # for w in [0.6, 0.7, 0.8, 0.9, 1]:
    #     test_multi(model, model2, model3, data_loader, w)
    
    for checkpoint in glob.glob("checkpoints/asl300/*.pth"):
        print('LOADING WEIGHTS...')
        print(checkpoint)
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint)
        print('WEIGHTS LOADED')
        test(model, data_loader)

    # for checkpoint in checkpoints:
    #     print('LOADING WEIGHTS...')
    #     print(checkpoint)
    #     checkpoint = torch.load(checkpoint)
    #     model.load_state_dict(checkpoint)
    #     print('WEIGHTS LOADED')
    #     test(model, data_loader)
    # print('LOADING WEIGHTS...')
    # print(checkpoints[0])
    # checkpoint = torch.load(root / f'code/TGCN/archived/{checkpoints[0]}')
    # model.load_state_dict(checkpoint)
    # print('WEIGHTS LOADED')

    # test_stacked_gru(model, data_loader)

    # model = StackedLSTM(input_size=110, hidden_size=256, num_classes=100, p_dropout=0.2).cuda()
    # print('LOADING WEIGHTS...')
    # print(checkpoints[1])
    # checkpoint = torch.load(root / f'code/TGCN/archived/{checkpoints[1]}')
    # model.load_state_dict(checkpoint)
    # print('WEIGHTS LOADED')

    # test_stacked_gru(model, data_loader)

    # print('LOADING WEIGHTS...')
    # checkpoint = 'checkpoints/asl100/stacked_gru-epoch=65-val_acc=0.5088757396449705-hidden=512.pth'
    # print(checkpoint)
    # checkpoint = torch.load(checkpoint)
    # model.load_state_dict(checkpoint)
    # print('WEIGHTS LOADED')
    # test_stacked_gru(model, data_loader)
