import argparse
import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm
from src.dataset_ucf import UCF_dataset
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
# from dataset import TSNDataSet
from src.models import TSN
from src.transforms import *
from src.opts import parser
from tqdm import tqdm
from src.autoaugment import ImageNetPolicy
best_prec1 = 0

class AverageMeter(object):
        """Computes and stores the average and current value"""
        def __init__(self):
            self.reset()

        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0

        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

class Trainer(object):
    def __init__(self, **args):
        for key in args:
            setattr(self, key, args[key])

class TSN_training(Trainer):

    def __init__(self, **args):
        super(TSN_training, self).__init__(**args)

    def get_training_object(self, num_class):
        #loading model
        model = TSN(num_class, self.num_segments, self.modality,
                    base_model=self.model_type,
                    consensus_type=self.consensus_type, dropout = self.dropout, partial_bn=not self.partialbn)

        # Checking device
        if torch.cuda.is_available():
            if self.gpus == "multi":
                device = torch.device("cuda")
                model = model.to(device)
                model = nn.DataParallel(model) 
            else:
                device = torch.device(self.gpus)
                model = model.to(device)
        else:
            device = torch.device("cpu")
            model  = model.to(device)

        #get policies and augmentation for training
        policies = model.get_optim_policies()
        train_augmentation = model.get_augmentation()

        transform = {"train": torchvision.transforms.Compose([
                        torchvision.transforms.Resize((self.size, self.size)),
                        ImageNetPolicy(),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ]),
                    "valid": torchvision.transforms.Compose([
                        torchvision.transforms.Resize((self.size, self.size)),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
        }
        # optimizer
        for group in policies:
            print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
                group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))
        optimizer = torch.optim.SGD(policies,
                                    self.lr,
                                    momentum=self.momentum,
                                    weight_decay=self.weight_decay)

        # Loss function 
        if self.loss_type == 'nll':
            criterion = torch.nn.CrossEntropyLoss().to(device)
        else:
            raise ValueError("Unknown loss type")

        return model, transform, optimizer, criterion, device

    def training(self, train_loader, model, criterion, optimizer, epoch, device):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        #Freezing BatchNorm2D except the first one if partialBN == True
        if self.partialbn:
            model.partialBN(True)
        else:
            model.partialBN(False)

        # switch to train mode
        model.train()

        end = time.time()
        for i, (inputs, targets) in enumerate(train_loader):

            # measure data loading time
            data_time.update(time.time() - end)

            input_var = inputs.to(device)
            
            # targets = torch.autograd.Variable(targets)
            target_var = targets.to(device)
            target_var =target_var.squeeze(1)

            # compute output
            output = model(input_var)
            # print("###############################")
            # print(target_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = self.accuracy(output, target_var, topk=(1,5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1, inputs.size(0))
            top5.update(prec5, inputs.size(0))


            # compute gradient and do SGD step
            optimizer.zero_grad()

            loss.backward()

            if self.clip_grad is not None:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad)
                if total_norm > self.clip_grad:
                    print("clipping gradient: {} with coef {}".format(total_norm, self.clip_grad / total_norm))

            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.print_freq == 0:
                print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'])))
            del input_var, target_var 

    def _test(self, test_loader, model, device):
        # output prediction and true label
        all_pred = []
        all_label = []
        model.eval()
        with torch.no_grad():
            for X, y in tqdm(test_loader):
                X = X.to(device)
                y = y.to(device)

                output = model(X)
                _, output = torch.max(output, 1)
                all_pred.extend(output)
                all_label.extend(y)

        all_label = torch.stack(all_label, dim = 0)
        all_label = all_label.cpu().squeeze().numpy()
        all_pred = torch.stack(all_pred, dim =0)
        all_pred = all_pred.cpu().squeeze().numpy()

        recall = recall_score(all_label, all_pred, average = 'macro')
        precision = precision_score(all_label, all_pred, average = 'macro')
        f1 = f1_score(all_label, all_pred, average = 'macro')
        test_score = accuracy_score(all_label, all_pred)
        print('\nTest set ({:d} samples): Accuracy: {:.2f}%\n, Recall: {:.2f}%\n, Precision: {:.2f}%\n, F1 score: {:.2f}%\n'.format(len(all_label), 100* test_score, 100*recall, 100*precision, 100*f1))
        del all_label, all_pred
        return test_score

    def validate(self, val_loader, model, criterion,  iter, device, logger=None):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        model.eval()

        end = time.time()
        for i, (inputs, target) in enumerate(val_loader):
            target_var = target.to(device)
            target_var =target_var.squeeze(1)
            input_var = inputs.to(device)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = self.accuracy(output, target_var, topk=(1,5))

            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1, inputs.size(0))
            top5.update(prec5, inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.print_freq == 0:
                print(('Valid: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5)))

        print(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
            .format(top1=top1, top5=top5, loss=losses)))
        del input_var, target_var, output 
        return top1.avg


    def save_checkpoint(self, state, is_best, epoch,  filename='checkpoint.pth'):
        best_folder = "best_weight"

        # Hot test on test-set
        if not os.path.exists(os.path.join(self.weight_model, best_folder)):
            os.mkdir(os.path.join(self.weight_model, best_folder))

        # folder for saving model after N epoch. Default: N = 4
        
        if not os.path.exists(os.path.join(self.weight_model, self.model_type)):
            os.mkdir(os.path.join(self.weight_model, self.model_type))

        filename = '_'.join((self.modality.lower(),str(epoch+1), filename))
        torch.save(state, os.path.join(self.weight_model, self.model_type,filename))
        if is_best:
            best_name = '_'.join((self.modality.lower(),'model_best.pth'))
            torch.save(state, os.path.join(self.weight_model, best_folder, best_name))

    def adjust_learning_rate(self, optimizer, epoch, lr_steps):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
        lr = self.lr * decay
        decay = self.weight_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * param_group['lr_mult']
            param_group['weight_decay'] = decay * param_group['decay_mult']


    def accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        # print(pred.size())
        # print(pred)
        pred = pred.t()
        temp_target = target.view(1, -1).expand_as(pred)
        # print(temp_target)
        correct = pred.eq(temp_target)
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        # print(res)
        return res
    
    def train(self):
        global  best_prec1

    ################## Loading model, optmizer,loss ###################################
        cudnn.benchmark = True

        # # Data loading code
        # if args.modality != 'RGBDiff':
        #     normalize = GroupNormalize(input_mean, input_std)
        # else:
        #     normalize = IdentityTransform()

        # if args.modality == 'RGB':
        #     data_length = 1
        # elif args.modality in ['Flow', 'RGBDiff']:
        #     data_length = 5

        # checking dataset
        if self.dataset == 'ucf101':
            num_class = 101
        elif self.dataset == 'hmdb51':
            num_class = 51
        elif self.dataset == 'kinetics':
            num_class = 400
        else:
            raise ValueError('Unknown dataset ' + self.dataset)

        model, transform, optimizer, criterion, device = self.get_training_object(num_class)

    ####################### preprocessing data ###########################################
        action = {}
        file1 = open(self.class_label, "r").readlines()
        label_pre = [x.split('\n')[0] for x in file1]
        label_pre = [x.split(' ') for x in label_pre]
        for label in label_pre:
            action[label[1]] = int(label[0])

        # get filename, label for train_set
        all_names = []
        label = []
        folder_name = os.listdir(self.data_path)
        test_sets = ["g01", "g02", "g03", "g04", "g05", "g06", "g07"]

        for f in folder_name:
            if f[-7:-4] not in test_sets:
                loc1 = f.find('v_')
                loc2 = f.find('_g')
                temp_act = f[(loc1+2) : loc2]
                label.append(action[temp_act])
                all_names.append(f)        

        X_train, X_val, y_train, y_val = train_test_split(all_names, label, stratify = label,  test_size = self.valid_size, random_state = 42)
        print("Total number of training samples: ", len(X_train))
        print("Total number of valid samples: ", len(X_val))

        #get filename and label for testset
        X_test = []
        y_test = []

        for f in folder_name:
            if f[-7:-4] in test_sets:
                loc1 = f.find('v_')
                loc2 = f.find('_g')
                temp_act = f[(loc1+2) : loc2]
                y_test.append(action[temp_act])
                X_test.append(f)        

        print("total videos in testset: ", len(X_test))

        """
        Config train_loader and valid_loader

        """

        train_set = UCF_dataset(self.data_path, X_train, y_train, num_segment= self.num_segments, transform = transform['train'])
        train_loader = data.DataLoader(train_set, batch_size = self.batch_size, shuffle = True, num_workers = self.num_worker, pin_memory = False)

        valid_set = UCF_dataset(self.data_path, X_val, y_val, num_segment= self.num_segments, transform = transform['valid'])
        val_loader = data.DataLoader(valid_set, batch_size = self.batch_size, shuffle = False, num_workers = self.num_worker, pin_memory = False)

        test_set = UCF_dataset(self.data_path, X_test, y_test, num_segment=self.num_segments, transform = transform['valid'])
        test_loader = data.DataLoader(test_set, batch_size= self.batch_size, shuffle=False, num_workers = self.num_worker, pin_memory= False)

    ##################################################################################################################################################

        for epoch in range(self.epochs):
            self.adjust_learning_rate(optimizer, epoch, self.lr_steps)

            # train for one epoch
            self.training(train_loader, model, criterion, optimizer, epoch, device = device)
            prec1 = self.validate(val_loader, model, criterion, (epoch + 1) * len(train_loader), device = device)
            # evaluate on validation set
            if (epoch + 1) % self.eval_freq == 0 or epoch == self.epochs - 1:
                score = self._test(test_loader, model, device= device)
                # remember best prec@1 and save checkpoint
                is_best = score > best_prec1
                best_prec1 = max(prec1, score)
                self.save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': self.model_type,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                }, is_best, epoch)
