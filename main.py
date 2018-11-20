import argparse
import os
import pdb
import time
import torch
import shutil

from taxiDataset import *

best_loss = np.Inf

def main():
    
    global best_loss, args
    
    parser = argparse.ArgumentParser(description="Main file for taxi prediction")
    
    # ========================= Model Configs ========================== 
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--loss', type=str, default='mae', choices=['mse', 'mae'])

    # ========================= Learning Configs ==========================
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_steps', type=int, default=[20, 40], help='epochs to decay learning rate by 10')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4) 

    # ========================= Monitor Configs ==========================
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-freq', type=int, default=5)

    # ========================= Runtime Configs ==========================
    parser.add_argument('--val', type=bool, default=False)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--iter_p_epoch', type=int, default=500)

    args = parser.parse_args()

    # Model architecture params
    feature_num = 3
    H1 = 4 
    H2 = 6 
    H3 = 5
    output_dim = 1

    # Model configuration
    model = torch.nn.Sequential(
        torch.nn.Linear(feature_num, H1),
        torch.nn.ReLU(),
        torch.nn.Linear(H1, H2),
        torch.nn.ReLU(),
        #torch.nn.Linear(H2, H3),
        #torch.nn.ReLU(),
        torch.nn.Linear(H2, output_dim),
        torch.nn.ReLU(),
    ).cuda()
   
    # Parameter Initialization
    model.apply(init_weights)

    # Load data
    dl = ['PULocationID','day_of_week','t_bucket']
    taxi_dataset = taxiDataset(csv_file='yellow_tripdata_2016-12.csv', desired_labels=dl)#, length=4000)
    train_loader = torch.utils.data.DataLoader(taxi_dataset.train, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    print(len(taxi_dataset))
    val_loader = torch.utils.data.DataLoader(taxi_dataset.val, batch_size=1, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(taxi_dataset.test, batch_size=1, shuffle=True, pin_memory=True)

    if args.val:
        if args.resume:
            if os.path.isfile(args.resume):
                print(("=> loading checkpoint '{}'".format(args.resume)))
                checkpoint = torch.load(args.resume)
                model_state_dict = checkpoint['model_state_dict']

                model.load_state_dict(model_state_dict)
                print("=> loaded checkpoint")
                
                validate(test_loader, model)
            else:
                raise ValueError(("=> no checkpoint found at '{}'".format(args.resume)))
        else:
            raise ValueError("Lack trained model as input. Use --resume")
    else:          
        # Load checkpoint, if any
        if args.resume:
            if os.path.isfile(args.resume):
                print(("=> loading checkpoint '{}'".format(args.resume)))
                checkpoint = torch.load(args.resume)
                args.start_epoch = checkpoint['epoch']
                best_loss = checkpoint['best_loss']
                model_state_dict = checkpoint['model_state_dict']

                model.load_state_dict(model_state_dict)
                print(("=> loaded checkpoint (epoch {})"
                    .format(checkpoint['epoch'])))
            else:
                print(("=> no checkpoint found at '{}'".format(args.resume)))

        # Optimization
        if args.loss == 'mse':
            loss = torch.nn.MSELoss(size_average=True)
        elif args.loss == 'mae':
            loss = torch.nn.L1Loss(size_average=True)
        else:
            raise ValueError("Only MSE and MAE are supported while we got {}".format(args.loss))

        #optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        optimizer = torch.optim.Adam(model.parameters())

        # Print the mean of ground truth labels for comparison
        print("The mean of ground truth labels is {}".format(loader_mean(val_loader)))

        # Training
        for epoch in range(args.start_epoch, args.epochs):
            # Evaluation on validation set
            if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
                loss_val = validate(val_loader, model)

                is_best = loss_val > best_loss
                best_loss = min(loss_val, best_loss)

                print("Loss on validation set for epoch {} is {}. Current minimum is {}".format(epoch, loss_val, best_loss))

                save_checkpoint({
                    'epoch' : epoch + 1,
                    'model_state_dict' : model.state_dict(),
                    'best_loss' : best_loss,
                }, is_best)

                #adjust_learning_rate(optimizer, epoch, args.lr_steps)

            train(train_loader, model, loss, optimizer, epoch, args.iter_p_epoch)

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.0)

def validate(val_loader, model):
    model.eval()

    all_output = []
    all_label = []

    with torch.no_grad():
        for i, (input_val, label) in enumerate(val_loader):
            input_val = input_val.to(torch.float32).cuda()
            output = model(input_val)
            all_output.append(output.item())
            all_label.append(label.item())

        loss = mse_loss(all_output, all_label)

    print('Loss: {:.3f}'.format(loss))
    
    return loss

def train(train_loader, model, loss, optimizer, epoch, iter_p_epoch):
    model.train()

    train_len = len(train_loader) - 1

    for i in range(iter_p_epoch):
        if i % train_len == 0:
            if i != 0:
                train_iter.next()
            train_iter = iter(train_loader)

        input_val, label = train_iter.next()
        input_val = input_val.to(torch.float32).cuda()
        label = label.to(torch.float32).cuda()

        output = model(input_val).view(-1)

        backable_loss = loss(output, label)

        optimizer.zero_grad()
        backable_loss.backward()

        optimizer.step()

        #if i % args.print_freq:
        #    print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
        #        #'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #        'c_Loss {c_loss.val:.4f} ({c_loss.avg:.4f})\t'
        #        .format(
        #        epoch, i, len(source_loader), batch_time=batch_time,
        #        data_time=data_time, c_loss=c_losses, ad_loss=ad_losses, lr=optimizer.param_groups[-1]['lr'])))

def mse_loss(output, label):
    return np.sum((np.asarray(output) - np.asarray(label)) ** 2) / len(output) 

def mae_loss(output, label):
    return np.sum(np.abs(np.asarray(output) - np.asarray(label))) / len(output)

def save_checkpoint(state, is_best, file_name='checkpoint.pth.tar'):
    torch.save(state, file_name)
    if is_best:
        best_name = 'model_best.pth.tar'
        shutil.copyfile(file_name, best_name)

def adjust_learning_rate(optimizer, epoch, lr_steps):
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']

def loader_mean(data_loader):
    mean = 0
    all_label = []

    with torch.no_grad():
        for i, (input_val, label) in enumerate(data_loader):
            mean += label.item()
            all_label.append(label)        
        mean /= len(all_label)

    return mean

if __name__ == '__main__':
    main()
