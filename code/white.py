import argparse
import os
import pickle
from fmnist_dataset import load_fashion_mnist
from model import CNN

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random

import numpy as np
import cv2

def attack(init_img, model, device=torch.device("cpu"), max_step = 1000, lr = 1):
 
    img = torch.tensor(init_img, device=device, requires_grad=True)
    #print(img)
    init_y = model(img)
    init_label = init_y.argmax()
    p_label, atk_label = init_y.kthvalue(k=2, dim=-1)

    for i in range(max_step):
        if img.grad is not None: img.grad.zero_()
        with torch.no_grad():
            img.clamp_(min = 0, max=255)
        logits = model(img)
        label = logits.argmax()
        #sprint("label current: %d, label desired: %d" % (label, atk_label))
        if label == atk_label or label != init_label:
            print("attack success at step %d, init label: %d current label: %d desired label %d" % 
                (i, init_label, label, atk_label))
            return True, img, label
        loss = F.cross_entropy(logits, atk_label)
        loss.backward()
        #print(img.grad)
        with torch.no_grad():   # custom arithmetic operation is not allowed on tensors with requires_grad=True
            img -= img.grad * lr
    return False, img, label

# def generate_data_for_adversarial_training(imgs, labels):


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='-1')
    parser.add_argument('--save_path', type=str, default='model/adv_cnn.pt')
    parser.add_argument('--dataset_dir', type=str, default='data')
    parser.add_argument('--eval_batch_size', type=int, default=1000)
    parser.add_argument('--rand_seed', type=int, default=42)
    parser.add_argument('--gen_data_path', type=str, default='attack_data/correct_1k.pkl')
    parser.add_argument('--gen_sample_path', type=str, default='attack_data/white_samples.pkl')
    parser.add_argument('--atk_img_path', type=str, default='attack_data/white_imgs')
    opt = parser.parse_args()
    
    if int(opt.gpu) < 0:
        device = torch.device('cpu')
        torch.manual_seed(opt.rand_seed)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
        device = torch.device("cuda")
        torch.manual_seed(opt.rand_seed)
        torch.cuda.manual_seed(opt.rand_seed)
    random.seed(opt.rand_seed)
    
    classifier = CNN().to(device)
    classifier.load_state_dict(torch.load(opt.save_path))
    for p in classifier.parameters():   # freeze parameters
        p.requires_grad = False
    
    with open(opt.gen_data_path, "rb") as f:
        imgs, labels = pickle.load(f)

    success_original_imgs = []
    success_attack_imgs = []
    success_attack_labels = []
    success_oringinal_labels = []

    for i, (img, label) in enumerate(zip(imgs, labels)):
        res, attack_img, attack_label = attack(img, classifier, device, max_step=1000, lr=1)
        if res:
            success_attack_imgs.append(attack_img.detach().cpu())
            success_attack_labels.append(attack_label)
            success_original_imgs.append(img)
            success_oringinal_labels.append(label)
            success_id = len(success_attack_imgs)
            print("success no. %d, total attempts: #%d" % (success_id, i))
            cv2.imwrite(os.path.join(opt.atk_img_path, "attack_%d_label_%d.png" % (success_id, attack_label)), 
                np.uint8(attack_img.detach().cpu()).reshape([28, 28]))
            cv2.imwrite(os.path.join(opt.atk_img_path, "origin_%d_label_%d.png" % (success_id, label)), 
                img.reshape([28, 28]))

    with open(opt.gen_sample_path, "wb") as f:
        pickle.dump([success_attack_imgs, success_oringinal_labels], f)
    # generate_data_for_adversarial_training
