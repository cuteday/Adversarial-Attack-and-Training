import argparse
import os
from mnist import MNIST
from fmnist_dataset import load_fashion_mnist, FashionMNISTDataset
from model import CNN

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import random
import pickle
    
    
def gettensor(x, y, device):
    return x.to(device), y.to(device)

def load_enhanced_data(fmnist_dir, adversarial_sample_path, n_dev=10000, random=None):
    fmnist = MNIST(fmnist_dir, return_type="lists")
    train = fmnist.load_training()
    test = fmnist.load_testing()
    
    assert n_dev >= 0 and n_dev <= len(train[0]), \
            "Invalid dev size %d, should be within 0 to %d" \
            % (n_dev, len(train[0]))
    if random is None:
        import random
    idx = random.sample(range(len(train[0])), len(train[0]))
    dev = [], []
    for i in idx[:n_dev]:
        dev[1].append(train[1][i])
        dev[0].append(train[0][i])
    _train = [], []
    for i in idx[n_dev:]:
        _train[1].append(train[1][i])
        _train[0].append(train[0][i])
    

    with open(adversarial_sample_path, "rb") as f:
        imgs, labels = pickle.load(f)
        print("loaded %d adversarial samples" % (len(imgs)))
        _train[0].extend(imgs)
        _train[1].extend(labels)

    return FashionMNISTDataset(_train), FashionMNISTDataset(dev), FashionMNISTDataset(test)
    
def trainEpochs(classifier, optimizer, loss_fn, epochs, training_set, dev_set,
                print_each, save_dir, device):
    
    for ep in range(1, epochs + 1):
        
        classifier.train()
        print_loss_total = 0
        
        print ('Ep %d' % ep)
        
        for i, (x, y) in enumerate(training_set):
            
            optimizer.zero_grad()
            x, y = gettensor(x, y, device)
            logits = classifier(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            
            print_loss_total += loss.item()
            if (i + 1) % print_each == 0: 
                print_loss_avg = print_loss_total / print_each
                print_loss_total = 0
                print('    %.4f' % print_loss_avg)
                
        acc = evaluate(classifier, dev_set, device)
        print ('  dev acc = %.2f%%' % acc)
        torch.save(classifier.state_dict(),
                   os.path.join(save_dir, 'adv_ep_' + str(ep) + '_devacc_' + str(acc) + '_.pt'))
        
            
def evaluate(classifier, dataset, device):
    
    classifier.eval()
    testnum = 0
    testcorrect = 0
    
    for x, y in dataset:
        
        with torch.no_grad():
            x, y = gettensor(x, y, device)
            logits = classifier(x)
            res = torch.argmax(logits, dim=1) == y
            testcorrect += torch.sum(res)
            testnum += len(y)
    
    acc = float(testcorrect) * 100.0 / testnum
    return acc
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='-1')
    parser.add_argument('--save_dir', type=str, default='../model')
    parser.add_argument('--dataset_dir', type=str, default='../data')
    parser.add_argument('--adversarial_data', type=str, default='../attack_data/white_samples.pkl')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--eval_batch_size', type=int, default=1000)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--rand_seed', type=int, default=42)
    parser.add_argument('--log_per_step', type=int, default=100)
    args = parser.parse_args()
    
    opt = parser.parse_args()
    
    device = torch.device('cpu')
    torch.manual_seed(opt.rand_seed)
    if args.gpu != "-1":
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
        device = torch.device("cuda")
        torch.manual_seed(opt.rand_seed)
        torch.cuda.manual_seed(opt.rand_seed)
        
    random.seed(opt.rand_seed)
    
    train, dev, _ = load_enhanced_data("../data", opt.adversarial_data , random=random)
    train_dataloader = DataLoader(train, batch_size=opt.batch_size, drop_last=True, shuffle=True)
    dev_dataloader = DataLoader(dev, batch_size=opt.eval_batch_size)

    classifier = CNN().to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=opt.learning_rate)
    criterion = nn.CrossEntropyLoss()

    trainEpochs(classifier, optimizer, criterion, opt.num_epochs,
                train_dataloader, dev_dataloader,
                opt.log_per_step, opt.save_dir, device)
