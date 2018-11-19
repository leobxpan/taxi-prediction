import argparse
import os
import time
import torch
import shutil

def main():
    parser = argparse.ArgumentParser(description="Main file for taxi prediction")
    
    # ========================= Model Configs ========================== 
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'mae'])

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
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('iter_p_epoch', type=int, default=500)

    args = parser.parse_args()

    # Model architecture params
    feature_num = 3
    H1 = 5
    H2 = 5
    output_dim = 1

    # Model configuration
    model = torch.nn.Sequential(
        torch.nn.Linear(feature_num, H1),
        torch.nn.ReLU(),
        torch.nn.Linear(H1, H2),
        torch.nn.Relu(),
        torch.nn.Linear(H2, output_dim),
        torch.nn.Relu()
    )

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

    # Load data
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True)
    
    # Optimization
    if args.loss == 'mse':
        loss = torch.nn.MSELoss(size_average=True)
    elif args.loss == 'mae':
        loss = torch.nn.L1Loss(size_average=True)
    else:
        raise ValueError("Only MSE and MAE are supported while we got {}".format(args.loss))

    optimizer = torch.optim.SGD(model.parameters, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    for epoch in range(args.start_epoch, args.epochs):
        # Evaluation on validation set
        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
            loss = validate(val_loader, model)
            
            is_best = loss > best_loss
            best_loss = min(loss, best_loss)

            print("Accuracy on validation set for epoch {} is {}%. Current best is {}%".format(epoch, loss, best_loss))

            save_checkpoint({
                'epoch' : epoch + 1,
                'model_state_dict' : model.state_dict(),
                'best_loss' : best_loss,
            }, is_best)

            adjust_learning_rate(optimizer, epoch, args.lr_steps)
        
            train(train_loader, val_loader, model, loss, optimizer, epoch, args.iter_p_epoch)

def validate(val_loader, model):
    model.eval()

    all_output = []
    all_label = []

    with torch.no_grad():
        for i, (input_val, label) in enumerate(val_loader):
            output = model(input_val)
            all_output.append(output.item())
            all_label.append(label.item())

    loss = mse_loss(all_output, all_label)

    print('Loss on validation set: {:.3f}'.format(loss))

def train(train_loader, val_loader, model, loss, optimizer, epoch, iter_p_epoch):
    model.train()

    train_len = len(train_loader) - 1
    val_len = len(val_loader) - 1

    for i in range(iter_p_epoch):
        if i % train_len == 0:
            if i != 0:
                train_iter.next()
            train_iter = iter(train_loader)
        if i % val_len == 0:
            if i != 0:
                val_iter.next()
            val_iter = iter(val_loader)

        input_val, label = train_iter.next()
        input_val = input_val.cuda()
        label = label.cuda()

        output = model(input_val)

        loss = mse_loss(output, label)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        #if i % args.print_freq:
        #    print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
        #        #'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #        'c_Loss {c_loss.val:.4f} ({c_loss.avg:.4f})\t'
        #        .format(
        #        epoch, i, len(source_loader), batch_time=batch_time,
        #        data_time=data_time, c_loss=c_losses, ad_loss=ad_losses, lr=optimizer.param_groups[-1]['lr'])))

#def mse_loss(output, label):

def save_checkpoint(state, is_best, file_name='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        best_name = 'model_best.pth.tar'
        shutil.copyfile(filename, bestname)

def adjust_learning_rate(optimizer, epoch, lr_steps):
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']

if __name__ == '__main__':
    main()
