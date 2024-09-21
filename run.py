import os
import argparse
import pickle as pk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import random
from sklearn.metrics import classification_report

from model.STGCNWQ import STGCNWQ

def load_data():
    A = np.load(".\\data\\distance_matrix.npy")
    A = A.astype(np.float32)
    X = np.load(".\\data\\LakeErie_noaa_cokriging.npy").transpose((1, 2, 0))
    X = X.astype(np.float32)

    # Normalization using Z-score method
    means = np.mean(X, axis=(0, 2))
    stds = np.std(X, axis=(0, 2))
    X = (X - means.reshape(1, -1, 1)) / stds.reshape(1, -1, 1)
    threshold = (2.371 - means[0]) / stds[0]
    print(threshold)

    return A, X, means, stds, threshold

from torch.distributions import MultivariateNormal as MVN

def bmc_loss_md(pred, target, noise_var=torch.tensor(.8)):
    """Compute the Multidimensional Balanced MSE Loss (BMC) between `pred` and the ground truth `targets`.
    Args:
      pred: A float tensor of size [batch, d].
      target: A float tensor of size [batch, d].
      noise_var: A float number or tensor.
    Returns:
      loss: A float tensor. Balanced MSE Loss.
    """
    # flatten the pred and target
    pred = pred.view(-1, pred.shape[-1])
    target = target.view(-1, target.shape[-1])
    I = torch.eye(pred.shape[-1])
    logits = MVN(pred.unsqueeze(1), noise_var*I).log_prob(target.unsqueeze(0))  # logit size: [batch, batch]
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]))     # contrastive-like loss
    loss = loss * (2 * noise_var).detach()  # optional: restore the loss scale, 'detach' when noise is learnable 
    
    return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=3, alpha=0.80, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        input = input.view(-1)
        target = target.view(-1)
        #input = torch.where(input>2.371, torch.tensor(1), torch.tensor(0))
        #target = torch.where(target>2.371, torch.tensor(1), torch.tensor(0))
        pt = torch.sigmoid(input)
        alpha = self.alpha
        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
               (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave


def generate_dataset(X, num_timesteps_input, num_timesteps_output, threshold):
    """
    Takes node features for the graph and divides them into multiple samples
    along the time-axis by sliding a window of size (num_timesteps_input+
    num_timesteps_output) across it in steps of 1.
    :param X: Node features of shape (num_vertices, num_features,
    num_timesteps)
    :return:
        - Node features divided into multiple samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_input).
        - Node targets for the samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_output).
    """
    # Generate the beginning index and the ending index of a sample, which
    # contains (num_points_for_training + num_points_for_predicting) points
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[2] - (
                num_timesteps_input + num_timesteps_output) + 1)]

    # Save samples
    hist, features, target = [], [], []
    for i, j in indices:
        hist.append(X[:, :, i: i + num_timesteps_input].transpose(
                (0, 2, 1)))
        feature = X[:, 1:X.shape[1], i + num_timesteps_input]
        feature = np.expand_dims(np.array(feature), axis = 1)
        features.append(feature)
        target.append(X[:, 0, i + num_timesteps_input: j])
    target = np.where(np.array(target) > threshold, 1, 0)

    return [torch.from_numpy(np.array(hist)), torch.from_numpy(np.array(features))], \
            torch.from_numpy(np.array(target))
            


use_gpu = True
num_timesteps_input = 12
num_timesteps_output = 1

epochs = 30
batch_size = 50

gamma = 1
alpha = 0.80

parser = argparse.ArgumentParser(description='STGCN_WQ')
parser.add_argument('--enable-cuda', action='store_true',
                    help='Enable CUDA')
args = parser.parse_args()
args.device = None
if args.enable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')


def train_epoch(training_input, training_target, batch_size):
    """
    Trains one epoch with the given data.
    :param training_input: Training inputs of shape (num_samples, num_nodes,
    num_timesteps_train, num_features).
    :param training_target: Training targets of shape (num_samples, num_nodes,
    num_timesteps_predict).
    :param batch_size: Batch size to use during training.
    :return: Average loss for this epoch.
    """
    permutation = torch.randperm(training_input[0].shape[0])

    epoch_training_losses = []
    for i in range(0, training_input[0].shape[0], batch_size):
        net.train()
        optimizer.zero_grad()

        indices = permutation[i:i + batch_size]
        hist_batch, features_batch, y_batch = training_input[0][indices], training_input[1][indices], training_target[indices]
        hist_batch = hist_batch.to(device=args.device)
        features_batch = features_batch.to(device=args.device)
        y_batch = y_batch.to(device=args.device)

        out = net(A_wave, hist_batch, features_batch)
        loss = loss_criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        epoch_training_losses.append(loss.detach().cpu().numpy())
    return sum(epoch_training_losses)/len(epoch_training_losses)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    torch.manual_seed(1)
    setup_seed(1)
    start_time = time.time()

    A, X, means, stds, threshold = load_data()

    split_line1 = int(X.shape[2] * 0.6)
    split_line2 = int(X.shape[2] * 0.8)


    train_original_data = X[:, :, :split_line1]
    val_original_data = X[:, :, split_line1:split_line2]
    test_original_data = X[:, :, split_line2:]

    print("Train shape: ", train_original_data.shape)
    print("Val shape: ", val_original_data.shape)
    print("Test shape: ", test_original_data.shape)

    training_input, training_target = generate_dataset(train_original_data,
                                                       num_timesteps_input=num_timesteps_input,
                                                       num_timesteps_output=num_timesteps_output,
                                                       threshold=threshold)
    val_input, val_target = generate_dataset(val_original_data,
                                             num_timesteps_input=num_timesteps_input,
                                             num_timesteps_output=num_timesteps_output,
                                                       threshold=threshold)
    test_input, test_target = generate_dataset(test_original_data,
                                               num_timesteps_input=num_timesteps_input,
                                               num_timesteps_output=num_timesteps_output,
                                                       threshold=threshold)

    A_wave = get_normalized_adj(A)
    A_wave = torch.from_numpy(A_wave)

    A_wave = A_wave.to(device=args.device)

    net = STGCNWQ(A_wave.shape[0],
                training_input[0].shape[3],
                num_timesteps_input,
                num_timesteps_output).to(device=args.device)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
    #loss_criterion = bmc_loss_md
    loss_criterion = FocalLoss(gamma=gamma, alpha=alpha, reduction='elementwise_mean')
    #loss_criterion = bmc_loss_md

    training_losses = []
    validation_losses = []
    validation_maes = []
    best_val_loss = 1e6
    for epoch in range(epochs):
        loss = train_epoch(training_input, training_target,
                           batch_size=batch_size)
        training_losses.append(loss)

        # Run validation
        with torch.no_grad():
            net.eval()
            val_hist = val_input[0].to(device=args.device)
            val_features = val_input[1].to(device=args.device)
            val_target = val_target.to(device=args.device)

            out = net(A_wave, val_hist, val_features)
            val_loss = loss_criterion(out, val_target).to(device=args.device)
            # validation_losses.append(np.asscalar(val_loss.detach().numpy()))
            validation_losses.append(val_loss.detach().numpy().item())
            out_unnormalized = out.detach().cpu().numpy()*stds[0]+means[0]
            target_unnormalized = val_target.detach().cpu().numpy()*stds[0]+means[0]
            mae = np.mean(np.absolute(out_unnormalized - target_unnormalized))
            validation_maes.append(mae)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(net.state_dict(), "./checkpoints/STGCN_best.pth")
        print("Epoch:" + str(epoch)+'\n')
        print("Training loss: {}".format(training_losses[-1]))
        print("Validation loss: {}".format(validation_losses[-1]))
        print("Validation MAE: {}".format(validation_maes[-1]))
        

        checkpoint_path = "checkpoints/"
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        with open("checkpoints/losses.pk", "wb") as fd:
            pk.dump((training_losses, validation_losses, validation_maes), fd)
        plt.clf()
        plt.plot(training_losses, label="training loss")
        plt.plot(validation_losses, label="validation loss")
        plt.legend()
        plt.pause(0.1)
        plt.ioff()
    end_time = time.time()
    # Run test
    net.load_state_dict(torch.load("checkpoints/STGCN_best.pth"))
    test_pred = net(A_wave, test_input[0], test_input[1])
    nan_index = np.load("./data/nan_index.npy")
    # convert to 1D arrays
    test_pred = test_pred.detach().cpu().numpy()
    test_pred_list = test_pred.flatten()
    test_target = test_target.detach().cpu().numpy()
    test_target_list = test_target.flatten()
    test_pred_nan = test_pred_list[nan_index]
    test_target_nan = test_target_list[nan_index]
    test_pred_nan_class = np.where(test_pred_nan > 0, 1, 0)
    test_target_nan_class = np.where(test_target_nan > 0, 1, 0)
    test_pred_class = np.where(test_pred_list > 0, 1, 0)
    test_target_class = np.where(test_target_list > 0, 1, 0)

    print(classification_report(test_target_class, test_pred_class))
 
    # save the prediction and target
    dir = '.\\experiment\\'
    with open(dir+'Erie_NOAA_report.txt','a') as f:
        f.write('Model:'+ "STGCN" +'\n')
        training_time = end_time - start_time
        m, s = divmod(training_time, 60)
        h, m = divmod(m, 60)
        difftime = "%02dh%02dm%02ds" % (h, m, s)
        curtime = time.localtime(start_time)
        f.write('Gamma:'+str(gamma)+'; Alpha:'+str(alpha)+'\n')
        f.write('Expriment Time:'+time.strftime("%Y-%m-%d %H:%M:%S",curtime)+'\n')
        f.write('Training Time:'+ str(difftime) +'\n')
        f.write('Loss:'+str(loss_criterion)+'\n')
        f.write('Best parameters found on train set:\n')
        f.write('epochs:'+str(epochs)+'\n')
        f.write('batch_size:'+str(batch_size)+'\n')
        f.write('lr:'+str(1e-3)+'\n')
        f.write('Number of input:{}'.format(num_timesteps_input)+'\n')
        f.write('Number of output:{}'.format(num_timesteps_output)+'\n')
        f.write(str(net)+'\n')
        f.write(classification_report(test_target_class, test_pred_class))
        f.write('\n\n')
    
