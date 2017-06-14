from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import sys
import itertools
import logging
from dataset_mnist import *
from dataset_usps import *
from net_config import *
from optparse import OptionParser

# Training settings
parser = OptionParser()
parser.add_option('--config',
                  type=str,
                  help="net configuration",
                  default="usps2mnist.yaml")

(opts, args) = parser.parse_args(sys.argv)
config = NetConfig(opts.config)
kwargs = {'num_workers': 1, 'pin_memory': True} if config.use_cuda else {}
torch.manual_seed(config.seed)
if torch.cuda.is_available() == False:
    config.use_cuda = False
    print("invalid cuda access") 
if config.use_cuda:
    torch.cuda.manual_seed(config.seed)


def read(argv,config):
    print(config)
    if os.path.exists(config.log):
        os.remove(config.log)
    base_folder_name = os.path.dirname(config.log)
    if not os.path.isdir(base_folder_name):
        os.mkdir(base_folder_name)
    logging.basicConfig(filename=config.log, level=logging.INFO, mode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    logging.info("Let the journey begin!")
    logging.info(config)
    exec("train_dataset_a = %s(root=config.train_data_a_path, \
                                  num_training_samples=config.train_data_a_size, \
                                  train=config.train_data_a_use_train_data, \
                                  transform=transforms.ToTensor(), \
                                  seed=config.train_data_a_seed)" % config.train_data_a)
    train_loader_a = torch.utils.data.DataLoader(dataset=train_dataset_a, batch_size=config.batch_size, shuffle=True)

    exec("train_dataset_b = %s(root=config.train_data_b_path, \
                                 num_training_samples=config.train_data_b_size, \
                                 train=config.train_data_b_use_train_data, \
                                 transform=transforms.ToTensor(), \
                                seed=config.train_data_b_seed)" % config.train_data_b)
    train_loader_b = torch.utils.data.DataLoader(dataset=train_dataset_b, batch_size=config.batch_size, shuffle=True)

    exec("test_dataset_b = %s(root=config.test_data_b_path, \
                                num_training_samples=config.test_data_b_size, \
                                train=config.test_data_b_use_train_data, \
                                transform=transforms.ToTensor(), \
                                seed=config.test_data_b_seed)" % config.test_data_b)
    test_loader_b = torch.utils.data.DataLoader(dataset=test_dataset_b, batch_size=config.test_batch_size, shuffle=True)

    return train_loader_a, train_loader_b, test_loader_b, 
    
train_loader_a, train_loader_b, test_loader_b = read(sys.argv,config)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x_f = self.fc1(x.view(-1, 320))
        x = F.dropout(F.relu(x_f), training=self.training)
        x = self.fc2(x)
        return x_f, F.log_softmax(x)

class Discrimer(nn.Module):
    def __init__(self):
        super(Discrimer, self).__init__()
        self.fc1 = nn.Linear(50, 512)
        self.fc2 = nn.Linear(512, 2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)


model = Net()
model_src = Net()
critic = Discrimer()

if config.use_cuda:
    model.cuda()
    model_src.cuda()
    critic.cuda()

optimizer_d = optim.Adam(critic.parameters(), lr=config.lr)
optimizer_g = optim.Adam(model.parameters(), lr=config.lr)
print("load model...")
PATH = config.pretrained_path #'pytorch_model_usps2mnist'    
model.load_state_dict(torch.load(PATH)) #model for adapt
model_src.load_state_dict(torch.load(PATH))


def train(epoch):
    model.train()
    
    for batch_idx, ((data_src, target_src), (data, target)) in enumerate(itertools.izip(train_loader_a, train_loader_b)):
        if config.use_cuda:
            data_src, target_src = data_src.cuda(), target_src.cuda()
            data, target = data.cuda(), target.cuda()
        data_src, target_src = Variable(data_src), Variable(target_src)
        data, target = Variable(data), Variable(target)
       
        feat_src, output_src = model_src(data_src)
        feat, output = model(data)
        all_d_feat = torch.cat((feat_src,feat),0)
        all_d_score = critic(all_d_feat)
        all_d_label = torch.cat((Variable(torch.ones(all_d_score.size()[0]/2).long().cuda()),
                     Variable(torch.zeros(all_d_score.size()[0]/2).long().cuda())),0)
        #D loss
        domain_loss = F.nll_loss(all_d_score, all_d_label)
        ###domain accuracy###
        predict = torch.squeeze(all_d_score.max(1)[1])
        d_accu = (predict == all_d_label).float().mean()

        critic.zero_grad()
        model.zero_grad()
        domain_loss.backward(retain_variables=True)
        optimizer_d.step()
        
        #G loss
        gen_loss = F.nll_loss(all_d_score[all_d_score.size()[0]/2:,...], 
                   Variable(torch.ones(all_d_score.size()[0]/2).long().cuda()))
        
        model.zero_grad()
        critic.zero_grad()
        gen_loss.backward()
        optimizer_g.step()
        
        if batch_idx % config.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tG Loss: {:.6f}\tD Loss: {:.6f}\tD accu: {:.3f}'.format(
                epoch, batch_idx * len(data), len(train_loader_a.dataset),
                100. * batch_idx / len(train_loader_a), gen_loss.data[0],domain_loss.data[0],d_accu.data[0]))

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader_b:
        if config.use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        feat, output = model(data)
        target = torch.squeeze(target)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(test_loader_b) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader_b.dataset),
        100. * correct / len(test_loader_b.dataset)))


for epoch in range(1, config.epochs + 1):
    train(epoch)    
    test(epoch)


