from argparse import ArgumentParser
from torch import nn
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os


import models.models_api
from config import config, device
from models import models_api
from data import MyDataloader


def train(model, train_loader, n_epochs, config, device, save_model_path):
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.SGD((param for param in model.parameters() if param.requires_grad),
                                lr=config['learning_rate'],
                                momentum=0.9,
                                weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config['lr_period'], config['lr_decay'])
    writer = SummaryWriter()
    model = model.to(device)

    n_epochs, step = n_epochs, 0

    for epoch in range(n_epochs):
        loss_record = []
        batch_size_sum = 0
        os.system('nvidia-smi')  # 执行nvidia-smi命令查看GPU状态

        train_pbar = tqdm(train_loader, position=0, leave=True)

        for i, (x, y) in enumerate(train_pbar):
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y).sum()
            loss.backward()
            optimizer.step()
            loss_record.append(loss.detach().item())
            batch_size_sum += y.shape[0]

            train_pbar.set_description(f'Epoch [{epoch + 1}/{n_epochs}]')

        scheduler.step()
        mean_train_loss = sum(loss_record) / batch_size_sum
        writer.add_scalar('Loss/train', mean_train_loss, step)

        print(f'Epoch [{epoch + 1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}')

    torch.save(model.state_dict(), save_model_path)  # Save your best model
    return


def test(model, valid_loader, device, pretrained_path):
    correct_predictions = 0
    total_samples = 0

    # 加载预训练参数
    pretrained_weights = torch.load(pretrained_path, weights_only=True)
    model.load_state_dict(pretrained_weights)

    # 遍历测试集
    for data, label in valid_loader:

        output = model(data.to(device))
        probs = torch.nn.functional.softmax(output, dim=1)
        _, predicted = torch.max(probs, dim=1)
        correct_predictions += (predicted.cpu() == label).sum().item()
        total_samples += label.size(0)

    # 计算 Top-1 精度
    top1_accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    print(f'Top-1 Test Accuracy: {top1_accuracy:.4f}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=256)
    parser.add_argument('-n', '--n_epochs', type=int, default=15)
    parser.add_argument('-t', '--test_phase', action='store_true')
    parser.add_argument('-m','--model', action='store')
    parser.add_argument('--model_name', type=str, default='fasternet_finetune', help='your model save path')

    args = parser.parse_args()

    if args.model == 'fasternet':
        model = models_api.finetune_fasternet_t0(device)
    elif args.model == 'resnet':
        model = models.models_api.finetune_resnet_18(device)
    elif args.model == 'mobilenet':
        model = models.models_api.finetune_mobilenet_v2(device)
    else:
        raise ValueError("You entered a model that does not exist.")

    if args.test_phase:
        valid_loader = MyDataloader(args.batch_size).val_dataloader()
        test(model, valid_loader, device, f'model_ckpt/{args.model_name}')
    else:
        train_loader = MyDataloader(args.batch_size).train_dataloader()
        train(model, train_loader, args.n_epochs, config, device, f'model_ckpt/{args.model_name}.pth')