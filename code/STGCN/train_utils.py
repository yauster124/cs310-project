import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

num_samples = 64

def train(log_interval, model, train_loader, optimizer, epoch):
    # ONLY FOR NON-TGCN
    model.train()

    # set model as training mode
    losses = []
    scores = []
    train_labels = []
    train_preds = []

    N_count = 0  # counting total trained sample in one epoch
    for batch_idx, data in enumerate(train_loader):
        X, y, video_ids = data
        # distribute data to device
        X, y = X.cuda(), y.cuda().view(-1, )
        N_count += X.size(0)
        optimizer.zero_grad()

        X = torch.stack(torch.chunk(X, num_samples, 2))
        X = torch.permute(X, (1, 0, 2, 3))
        X = torch.flatten(X, start_dim=2)

        out = model(X)  # output has dim = (batch, number of classes)
        loss = compute_loss(out, y)
        losses.append(loss.item())

        # to compute accuracy
        y_pred = torch.max(out, 1)[1]  # y_pred != output
        step_score = accuracy_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())

        # collect prediction labels
        train_labels.extend(y.cpu().data.squeeze().tolist())
        train_preds.extend(y_pred.cpu().data.squeeze().tolist())
        scores.append(step_score)  # computed on CPU
        loss.backward()
        optimizer.step()

        # show information
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.6f}%'.format(
                epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(),
                100 * step_score))

    return losses, scores, train_labels, train_preds

def train_stacked_gru(log_interval, model, train_loader, optimizer, epoch):
    # ONLY FOR NON-TGCN
    model.train()

    # set model as training mode
    losses = []
    scores = []
    train_labels = []
    train_preds = []

    N_count = 0  # counting total trained sample in one epoch
    for batch_idx, data in enumerate(train_loader):
        # X: (b, n, k, f)
        X, y, _ = data
        X, y = X.cuda(), y.cuda().view(-1, )

        N_count += X.size(0)

        optimizer.zero_grad()

        # X = torch.stack(torch.chunk(X, num_samples, 2))
        # X = torch.permute(X, (1, 0, 2, 3))
        # X = torch.flatten(X, start_dim=2)
        print(X.shape)
        exit()
        
        # X: (b, n, k*f)
        out = model(X)  # output has dim = (batch, number of classes)

        loss = compute_loss(out, y)

        # loss = F.cross_entropy(output, y)
        losses.append(loss.item())

        # to compute accuracy
        y_pred = torch.max(out, 1)[1]  # y_pred != output

        step_score = accuracy_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())

        # collect prediction labels
        train_labels.extend(y.cpu().data.squeeze().tolist())
        train_preds.extend(y_pred.cpu().data.squeeze().tolist())

        scores.append(step_score)  # computed on CPU

        loss.backward()

        optimizer.step()

        # show information
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.6f}%'.format(
                epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(),
                100 * step_score))

    return losses, scores, train_labels, train_preds


def validation(model, test_loader, epoch, save_to):
    # set model as testing mode
    model.eval()

    val_loss = []
    all_y = []
    all_y_pred = []
    all_video_ids = []
    all_pool_out = []

    num_copies = 4

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            # distribute data to device
            X, y, video_ids = data
            X, y = X.cuda(), y.cuda().view(-1, )

            all_output = []

            # ONLY FOR NON GRU
            stride = X.size()[2] // num_copies

            for i in range(num_copies):
                # X_slice = X[:, :, i * stride: (i+1) * stride]
                # X_slice = torch.stack(torch.chunk(X_slice, num_samples, 2))
                # X_slice = torch.permute(X_slice, (1, 0, 2, 3))
                # X_slice = torch.flatten(X_slice, start_dim=2)
                output = model(X_slice)
                all_output.append(output)

            all_output = torch.stack(all_output, dim=1)
            output = torch.mean(all_output, dim=1)


            # ONLY FOR TGCN GRU
            # stride = 2
            # X = torch.stack(torch.chunk(X, 50, 2))
            # n, b, k, f = X.shape
            # X = X.view(b, n, k, f)
            # print(f'BEFORE: {X.shape}')
            # for i in range(num_copies):
            #     X_slice = X[:, :, :, i * stride: (i+1) * stride]
            #     print(f'AFTER: {X_slice.shape}')
            #     output = model(X_slice)
            #     all_output.append(output)
            # all_output = torch.stack(all_output, dim=1)
            # output = torch.mean(all_output, dim=1)

            # ONLY FOR STACKED GRU
            # X = torch.stack(torch.chunk(X, 50, 2))
            # n, b, k, f = X.shape
            # X = X.view(n, b, k*f)  
            # stride = 110
            # for i in range(num_copies):
            #     X_slice = X[:, :, i * stride: (i+1) * stride]
            #     output = model(X_slice)
            #     all_output.append(output)
            # all_output = torch.stack(all_output, dim=1)
            # output = torch.mean(all_output, dim=1)

            # ONLY FOR MP
            # stride = X.size()[2] // num_copies

            # for i in range(num_copies):
            #     X_slice = X[:, i * stride: (i+1) * stride, :]
            #     print(X_slice.shape)
            #     output = model(X_slice)
            #     all_output.append(output)

            # all_output = torch.stack(all_output, dim=1)
            # output = torch.mean(all_output, dim=1)

            # output = model(X)  # output has dim = (batch, number of classes)

            # loss = F.cross_entropy(pool_out, y, reduction='sum')
            loss = compute_loss(output, y)

            val_loss.append(loss.item())  # sum up batch loss
            y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(y_pred)
            all_video_ids.extend(video_ids)
            all_pool_out.extend(output)

    # this computes the average loss on the BATCH
    val_loss = sum(val_loss) / len(val_loss)

    # compute accuracy
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0).squeeze()
    all_pool_out = torch.stack(all_pool_out, dim=0).cpu().data.numpy()

    # log down incorrectly labelled instances
    incorrect_indices = torch.nonzero(all_y - all_y_pred).squeeze().data
    incorrect_video_ids = [(vid, int(all_y_pred[i].data)) for i, vid in enumerate(all_video_ids) if
                           i in incorrect_indices]

    all_y = all_y.cpu().data.numpy()
    all_y_pred = all_y_pred.cpu().data.numpy()

    # top-k accuracy
    top1acc = accuracy_score(all_y, all_y_pred)
    top3acc = compute_top_n_accuracy(all_y, all_pool_out, 3)
    top5acc = compute_top_n_accuracy(all_y, all_pool_out, 5)
    top10acc = compute_top_n_accuracy(all_y, all_pool_out, 10)
    top30acc = compute_top_n_accuracy(all_y, all_pool_out, 30)

    # show information
    print('\nVal. set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(all_y), val_loss,
                                                                                        100 * top1acc))

    if save_to:
        # save Pytorch models of best record
        torch.save(model.state_dict(),
                   os.path.join(save_to, 'gcn_epoch{}.pth'.format(epoch + 1)))  # save spatial_encoder
        print("Epoch {} model saved!".format(epoch + 1))

    return val_loss, [top1acc, top3acc, top5acc, top10acc, top30acc], all_y.tolist(), all_y_pred.tolist(), incorrect_video_ids

def validation_stacked_gru(model, test_loader, epoch, save_to):
    # set model as testing mode
    model.eval()

    val_loss = []
    all_y = []
    all_y_pred = []
    all_video_ids = []
    all_pool_out = []

    num_copies = 4

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            # distribute data to device
            X, y, video_ids = data
            X, y = X.cuda(), y.cuda().view(-1, )

            all_output = []

            # ONLY FOR NON GRU
            stride = X.size()[2] // num_copies

            for i in range(num_copies):
                X_slice = X[:, :, i * stride: (i+1) * stride]
                X_slice = torch.stack(torch.chunk(X_slice, 50, 2))
                X_slice = torch.permute(X_slice, (1, 0, 2, 3))
                X_slice = torch.flatten(X_slice, start_dim=2)
                output = model(X_slice)
                all_output.append(output)

            all_output = torch.stack(all_output, dim=1)
            output = torch.mean(all_output, dim=1)

            # output = model(X)  # output has dim = (batch, number of classes)

            # loss = F.cross_entropy(pool_out, y, reduction='sum')
            loss = compute_loss(output, y)

            val_loss.append(loss.item())  # sum up batch loss
            y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(y_pred)
            all_video_ids.extend(video_ids)
            all_pool_out.extend(output)

    # this computes the average loss on the BATCH
    val_loss = sum(val_loss) / len(val_loss)

    # compute accuracy
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0).squeeze()
    all_pool_out = torch.stack(all_pool_out, dim=0).cpu().data.numpy()

    # log down incorrectly labelled instances
    incorrect_indices = torch.nonzero(all_y - all_y_pred).squeeze().data
    incorrect_video_ids = [(vid, int(all_y_pred[i].data)) for i, vid in enumerate(all_video_ids) if
                           i in incorrect_indices]

    all_y = all_y.cpu().data.numpy()
    all_y_pred = all_y_pred.cpu().data.numpy()

    # top-k accuracy
    top1acc = accuracy_score(all_y, all_y_pred)
    top3acc = compute_top_n_accuracy(all_y, all_pool_out, 3)
    top5acc = compute_top_n_accuracy(all_y, all_pool_out, 5)
    top10acc = compute_top_n_accuracy(all_y, all_pool_out, 10)
    top30acc = compute_top_n_accuracy(all_y, all_pool_out, 30)

    # show information
    print('\nVal. set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(all_y), val_loss,
                                                                                        100 * top1acc))

    if save_to:
        # save Pytorch models of best record
        torch.save(model.state_dict(),
                   os.path.join(save_to, 'gcn_epoch{}.pth'.format(epoch + 1)))  # save spatial_encoder
        print("Epoch {} model saved!".format(epoch + 1))

    return val_loss, [top1acc, top3acc, top5acc, top10acc, top30acc], all_y.tolist(), all_y_pred.tolist(), incorrect_video_ids

def compute_loss(out, gt):
    ce_loss = F.cross_entropy(out, gt)

    return ce_loss


def compute_top_n_accuracy(truths, preds, n):
    best_n = np.argsort(preds, axis=1)[:, -n:]
    ts = truths
    successes = 0
    for i in range(ts.shape[0]):
        if ts[i] in best_n[i, :]:
            successes += 1
    return float(successes) / ts.shape[0]
