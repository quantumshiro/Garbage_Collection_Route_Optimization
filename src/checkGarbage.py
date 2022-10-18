import glob
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm.notebook import tqdm
from vit_pytorch.efficient import ViT
from sklearn.model_selection import train_test_split
from pathlib import Path
import seaborn as sns
import timm
from pprint import pprint
import shutil

epochs = 50
lr = 3e-5
gamma = 0.7
seed = 42
device = 'cuda'

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def image_dir_train_test_sprit(original_dir, base_dir, train_size=0.8):
    '''
    画像データをトレインデータとテストデータにシャッフルして分割します。フォルダもなければ作成します。

    parameter
    ------------
    original_dir: str
      オリジナルデータフォルダのパス その下に各クラスのフォルダがある
    base_dir: str
      分けたデータを格納するフォルダのパス　そこにフォルダが作られます
    train_size: float
      トレインデータの割合
    '''
    try:
        os.mkdir(base_dir)
    except FileExistsError:
        print(base_dir + "は作成済み")

    #クラス分のフォルダ名の取得
    dir_lists = os.listdir(original_dir)
    dir_lists = [f for f in dir_lists if os.path.isdir(os.path.join(original_dir, f))]
    original_dir_path = [os.path.join(original_dir, p) for p in dir_lists]

    num_class = len(dir_lists)

    # フォルダの作成(トレインとバリデーション)
    try:
        train_dir = os.path.join(base_dir, 'train')
        os.mkdir(train_dir)
    except FileExistsError:
        print(train_dir + "は作成済み")

    try:
        validation_dir = os.path.join(base_dir, 'validation')
        os.mkdir(validation_dir)
    except FileExistsError:
        print(validation_dir + "は作成済み")

    #クラスフォルダの作成
    train_dir_path_lists = []
    val_dir_path_lists = []
    for D in dir_lists:
        train_class_dir_path = os.path.join(train_dir, D)
        try:
            os.mkdir(train_class_dir_path)
        except FileExistsError:
            print(train_class_dir_path + "は作成済み")
        train_dir_path_lists += [train_class_dir_path]
        val_class_dir_path = os.path.join(validation_dir, D)
        try:
            os.mkdir(val_class_dir_path)
        except FileExistsError:
            print(val_class_dir_path + "は作成済み")
        val_dir_path_lists += [val_class_dir_path]


    #元データをシャッフルしたものを上で作ったフォルダにコピーします。
    #ファイル名を取得してシャッフル
    for i,path in enumerate(original_dir_path):
        files_class = os.listdir(path)
        random.shuffle(files_class)
        # 分割地点のインデックスを取得
        num_bunkatu = int(len(files_class) * train_size)
        #トレインへファイルをコピー
        for fname in files_class[:num_bunkatu]:
            src = os.path.join(path, fname)
            dst = os.path.join(train_dir_path_lists[i], fname)
            shutil.copyfile(src, dst)
        #valへファイルをコピー
        for fname in files_class[num_bunkatu:]:
            src = os.path.join(path, fname)
            dst = os.path.join(val_dir_path_lists[i], fname)
            shutil.copyfile(src, dst)
        print(path + "コピー完了")

    print("分割終了")
    
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    
    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

def main():
    # image dir train test split
    # image_dir_train_test_sprit(original_dir='data/image', base_dir='data/image/split_data', train_size=0.8)
    seed_everything(seed)
    
    train_dir = 'data/image/split_data/train'
    val_dir = 'data/image/split_data/validation'
    
    files = glob.glob('data/image/split_data/*/*/*.jpg')
    print('train data num: ', len(files))
    random_idx = np.random.randint(1, len(files), size=9)
    
    train_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    )
    
    val_transforms = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    )
    
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_data = datasets.ImageFolder(val_dir, transform=val_transforms)
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=True)
    
    # read Vit Model
    model_names = timm.list_models(pretrained=True)
    pprint(model_names)
    
    model = timm.create_model('vit_base_patch16_224_in21k', pretrained=True)
    model.to("cuda:0")
    
    # train model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=gamma)
    
    train_acc_list = []
    val_acc_list = []
    train_loss_list = []
    val_loss_list = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0

        for data, label in tqdm(train_loader):
            data,target_a,target_b,lam = mixup_data(data, label, alpha=1.0, use_cuda=torch.cuda.is_available())
            data = data.to(device)
            label = target_a.to(device)

            output = model(data)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)              

    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for data, label in val_loader:
            data = data.to(device)
            label = label.to(device)

            val_output = model(data)
            val_loss = criterion(val_output, label)

            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(val_loader)
            epoch_val_loss += val_loss / len(val_loader)

    print(
        f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
    )
    train_acc_list.append(epoch_accuracy)
    val_acc_list.append(epoch_val_accuracy)
    train_loss_list.append(epoch_loss)
    val_loss_list.append(epoch_val_loss)
        
    # save model
    # save dir -> model/model.pth
    if not os.path.exists('model'):
        os.mkdir('model')
    torch.save(model.state_dict(), 'model/model.pth')
    
    
if __name__ == '__main__':
    main()
    