import os
import time
import datetime
import torch
import torch.nn as nn
import glob
import os.path as osp
import numpy as np
import torch.nn.functional as F

from model import UNet
from model import Discriminator
from torchvision.utils import save_image
from evaluation import plotingConfusionMatrix
from evaluation import calculateROC
from opt_threshold import OptThreshold


class Solver(object):

    def __init__(self, config, data_loader_train, data_loader_test):
        """Initialize configurations."""
        self.data_loader_train = data_loader_train
        self.data_loader_test = data_loader_test

        self.epoch         = config['TRAINING_CONFIG']['EPOCH']
        self.batch_size    = config['TRAINING_CONFIG']['BATCH_SIZE']
        self.g_lr          = float(config['TRAINING_CONFIG']['G_LR'])
        self.d_lr          = float(config['TRAINING_CONFIG']['D_LR'])
        self.lambda_g_fake = config['TRAINING_CONFIG']['LAMBDA_G_FAKE']
        self.lambda_g_recon = config['TRAINING_CONFIG']['LAMBDA_G_RECON']
        self.lambda_d_fake = config['TRAINING_CONFIG']['LAMBDA_D_FAKE']
        self.lambda_d_real = config['TRAINING_CONFIG']['LAMBDA_D_REAL']
        self.lambda_d_gp     = config['TRAINING_CONFIG']['LAMBDA_GP']
        self.lambda_classification = config['TRAINING_CONFIG']['LAMBDA_CLASSIFICATION']
        self.d_critic      = config['TRAINING_CONFIG']['D_CRITIC']
        self.g_critic      = config['TRAINING_CONFIG']['G_CRITIC']
        self.mse_loss = nn.MSELoss()
        self.cross_entropy = nn.CrossEntropyLoss()

        self.gan_loss = config['TRAINING_CONFIG']['GAN_LOSS']
        assert self.gan_loss in ['lsgan', 'wgan', 'vanilla']

        self.optim = config['TRAINING_CONFIG']['OPTIM']
        self.beta1 = config['TRAINING_CONFIG']['BETA1']
        self.beta2 = config['TRAINING_CONFIG']['BETA2']

        if self.gan_loss == 'lsgan':
            self.adversarial_loss = torch.nn.MSELoss()
        elif self.gan_loss =='vanilla':
            self.adversarial_loss = torch.nn.BCELoss()
        self.l1_loss = torch.nn.L1Loss()

        self.cpu_seed = config['TRAINING_CONFIG']['CPU_SEED']
        self.gpu_seed = config['TRAINING_CONFIG']['GPU_SEED']
        #torch.manual_seed(config['TRAINING_CONFIG']['CPU_SEED'])
        #torch.cuda.manual_seed_all(config['TRAINING_CONFIG']['GPU_SEED'])

        self.g_spec = config['TRAINING_CONFIG']['G_SPEC'] == 'True'
        self.d_spec = config['TRAINING_CONFIG']['D_SPEC'] == 'True'

        self.gpu_param = config['TRAINING_CONFIG']['GPU']

        if self.gpu_param != '':
            self.gpu_list = [int(gpu) for gpu in self.gpu_param.split(',')]
            self.num_gpu = len(self.gpu_list)
            if len(self.gpu_list):
                self.gpu = torch.device('cuda:' + str(self.gpu_list[0]))
            print('num_gpu : ', self.num_gpu)
            print('gpu : ', self.gpu)
        else:
            self.num_gpu = 0
            self.gpu = None
            self.gpu_list = None

        self.use_tensorboard = config['TRAINING_CONFIG']['USE_TENSORBOARD']

        # Directory
        self.train_dir  = config['TRAINING_CONFIG']['TRAIN_DIR']
        self.log_dir    = os.path.join(self.train_dir, config['TRAINING_CONFIG']['LOG_DIR'])
        self.sample_dir = os.path.join(self.train_dir, config['TRAINING_CONFIG']['SAMPLE_DIR'])
        self.result_dir = os.path.join(self.train_dir, config['TRAINING_CONFIG']['RESULT_DIR'])
        self.model_dir  = os.path.join(self.train_dir, config['TRAINING_CONFIG']['MODEL_DIR'])

        # Steps
        self.log_step       = config['TRAINING_CONFIG']['LOG_STEP']
        self.sample_step    = config['TRAINING_CONFIG']['SAMPLE_STEP']
        self.save_step      = config['TRAINING_CONFIG']['SAVE_STEP']
        self.save_start     = config['TRAINING_CONFIG']['SAVE_START']
        self.lr_decay_step  = config['TRAINING_CONFIG']['LR_DECAY_STEP']

        self.build_model()

        if self.use_tensorboard == 'True':
            self.build_tensorboard()

    def build_model(self):

        if self.num_gpu > 1:
            print("Multi-gpu use")
            self.G = nn.DataParallel(UNet(in_channels=3, out_channels=3, spec_norm=self.g_spec, LR=0.02), device_ids=self.gpu_list).to(self.gpu)
            self.D = nn.DataParallel(Discriminator(in_channel=3, spec_norm=self.d_spec, LR=0.02), device_ids=self.gpu_list).to(self.gpu)
        elif self.num_gpu == 1:
            self.G = UNet(in_channels=3, out_channels=3, spec_norm=self.g_spec, LR=0.02).to(self.gpu)
            self.D = Discriminator(in_channel=3, spec_norm=self.d_spec, LR=0.02).to(self.gpu)
        else:
            self.G = UNet(in_channels=3, out_channels=3, spec_norm=self.g_spec, LR=0.02)
            self.D = Discriminator(in_channel=3, spec_norm=self.d_spec, LR=0.02)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, (self.beta1, self.beta2))
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, (self.beta1, self.beta2))

        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        #print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

        with open(os.path.join(self.train_dir,'model_arch.txt'), 'a') as fp:
            print(model, file=fp)
            print(name, file=fp)
            print("The number of parameters: {}".format(num_params),file=fp)

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.gpu)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

    def restore_model(self):

        ckpt_list = glob.glob(osp.join(self.model_dir, '*-G.ckpt'))

        if len(ckpt_list) == 0:
            return 0

        ckpt_list = [int(x[0]) for x in ckpt_list]
        ckpt_list.sort()
        epoch = ckpt_list[-1]
        G_path = os.path.join(self.model_dir, '{}-G.ckpt'.format(epoch))
        D_path = os.path.join(self.model_dir, '{}-D.ckpt'.format(epoch))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
        self.G.to(self.gpu)
        self.D.to(self.gpu)
        return epoch

    """
    prediction = self.model(train_images)
    # prediction shape is (batch_size , numClass)
    # target shape is (batch_size)
    traingLoss = self.criterion(prediction, target) * self.lambdaLoss

    _, prd_idx = torch.max(prediction, 1)
    correct = (prd_idx == target).sum().cpu().item()

    """
    def train(self):

        # Set data loader.
        data_loader = self.data_loader_train
        iterations = len(self.data_loader_train)
        print('iterations : ', iterations)
        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        _, _, fixed_train_images, fixed_target = next(data_iter)

        fixed_train_images = fixed_train_images.to(self.gpu)
        fixed_target = fixed_target.to(self.gpu)
        fixed_target = fixed_target.squeeze()

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        start_epoch = self.restore_model()
        start_time = time.time()
        print('Start training...')
        for e in range(start_epoch, self.epoch):

            for i in range(iterations):
                try:
                    _, _, train_images, target = next(data_iter)
                    target = target.squeeze()
                except:
                    data_iter = iter(data_loader)
                    _, _, train_images, target = next(data_iter)
                    target = target.squeeze()

                train_images = train_images.to(self.gpu)
                target = target.to(self.gpu)

                loss_dict = dict()
                if (i + 1) % self.d_critic == 0:
                    fake_images, _ = self.G(train_images)
                    d_loss = None

                    real_score, d_pred = self.D(train_images)
                    fake_score, _ = self.D(fake_images.detach())
                    d_classification = self.cross_entropy(d_pred, target)

                    if self.gan_loss in ['lsgan', 'vanilla']:
                        d_loss_real = self.adversarial_loss(real_score, torch.ones_like(real_score))
                        d_loss_fake = self.adversarial_loss(fake_score, torch.zeros_like(fake_score))
                        d_loss = self.lambda_d_real * d_loss_real + self.lambda_d_fake * d_loss_fake + self.lambda_classification * d_classification
                    elif self.gan_loss == 'wgan':
                        d_loss_real = -torch.mean(real_score)
                        d_loss_fake = torch.mean(fake_score)
                        alpha = torch.rand(train_images.size(0), 1, 1, 1).to(self.gpu)
                        x_hat = (alpha * train_images.data + (1 - alpha) * fake_images.data).requires_grad_(True)
                        out_src, _ = self.D(x_hat)
                        d_loss_gp = self.gradient_penalty(out_src, x_hat)
                        d_loss = self.lambda_d_real * d_loss_real + self.lambda_d_fake * d_loss_fake + self.lambda_d_gp * d_loss_gp + self.lambda_classification * d_classification

                    # Backward and optimize.
                    self.reset_grad()
                    d_loss.backward()
                    self.d_optimizer.step()

                    # Logging.
                    loss_dict['D/loss_real'] = self.lambda_d_real * d_loss_real.item()
                    loss_dict['D/loss_fake'] = self.lambda_d_fake * d_loss_fake.item()
                    loss_dict['D/loss_classification'] = self.lambda_classification * d_classification.item()

                    if self.gan_loss == 'wgan':
                        loss_dict['D/loss_pg'] = self.lambda_d_gp * d_loss_gp.item()

                if (i + 1) % self.g_critic == 0:
                    fake_images, prediction = self.G(train_images)
                    fake_score, _ = self.D(fake_images)
                    if self.gan_loss in ['lsgan', 'vanilla']:
                        g_loss_fake = self.adversarial_loss(fake_score, torch.ones_like(fake_score))
                    elif self.gan_loss == 'wgan':
                        g_loss_fake = - torch.mean(fake_score)

                    g_loss_recon = self.l1_loss(fake_images, train_images)
                    classiciation_loss = self.cross_entropy(prediction, target)
                    g_loss = self.lambda_g_fake * g_loss_fake + self.lambda_g_recon + g_loss_recon + self.lambda_classification * classiciation_loss

                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    # Logging.
                    loss_dict['G/loss_fake'] = self.lambda_g_fake * g_loss_fake.item()
                    loss_dict['G/loss_recon'] = self.lambda_g_recon * g_loss_recon.item()
                    loss_dict['G/loss_classification'] = self.lambda_classification * classiciation_loss.item()

                if (i + 1) % self.log_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Epoch [{}/{}], Elapsed [{}], Iteration [{}/{}]".format(e+1, self.epoch, et, i + 1, iterations)
                    for tag, value in loss_dict.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

            if (e + 1) % self.sample_step == 0:
                with torch.no_grad():
                    image_report_compare = list()
                    image_report_fake = list()

                    fixed_fake_images, _ = self.G(fixed_train_images)
                    image_report_compare.append(fixed_train_images)
                    image_report_compare.append(fixed_fake_images)
                    x_concat = torch.cat(image_report_compare, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(e + 1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)

                    image_report_fake.append(fixed_fake_images)
                    x_concat = torch.cat(image_report_fake, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images_fake.jpg'.format(e + 1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(self.sample_dir))
                self.testing(e + 1, self.data_loader_test)
            # Save model checkpoints.
            if (e + 1) % self.save_step == 0 and (e + 1) >= self.save_start:
                G_path = os.path.join(self.model_dir, '{}-G.ckpt'.format(e + 1))
                D_path = os.path.join(self.model_dir, '{}-D.ckpt'.format(e + 1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_dir))

        print('Training is finished')

    def testing(self, epoch, data_loader):
        # Set data loader.
        test_dataloader = data_loader
        class_list = ['normal', 'abnormal']
        # https://discuss.pytorch.org/t/model-train-and-model-eval-vs-model-and-model-eval/5744
        self.G = self.G.eval()

        with torch.no_grad():
            correct = 0
            total = 0
            pred_list = list()
            pred_idx_list = list()
            target_list = list()
            testing_loss_list = list()

            for i, data in enumerate(test_dataloader):
                _, _, test_images, target = data
                test_images = test_images.to(self.gpu)
                target = target.to(self.gpu)
                target = target.squeeze()

                _, prediction = self.G(test_images)
                testing_loss = self.cross_entropy(prediction, target.unsqueeze(0)) * self.lambda_classification
                testing_loss_list.append(testing_loss.item())
                pred_list.append(prediction)
                target_list.append(target)
                _, prediction = torch.max(prediction, 1)
                pred_idx_list.append(prediction)
                total += 1
                correct += (prediction == target).sum().item()

            testing_loss = np.mean(testing_loss_list)
            test_accuracy = (100 * correct / total)

            print('Accuracy of the net on the test dataset, : {:.4f}'.format(test_accuracy))
            print('loss of the net on the test dataset, : {:.4f}'.format(testing_loss))

            pred_soft_numpy = F.softmax(torch.cat(pred_list, dim=0),
                                        dim=1).cpu().numpy()  # shape : [batch_size, numClass]
            pred_idx_numpy = torch.stack(pred_idx_list).cpu().numpy()  # shape : [batch_size, 1]
            pred_idx_numpy = np.squeeze(pred_idx_numpy)  # shape : [batch_size]
            target_numpy = torch.stack(target_list).cpu().numpy()  # shape : [batch_size]

            results = list()
            # this is to make 2 classes probability into one (just in case of different activation functions)
            for i in range(len(pred_soft_numpy)):
                # print(predictions[i])
                if pred_soft_numpy[i][0] < pred_soft_numpy[i][1]:
                    results.append([pred_soft_numpy[i][1]])
                else:
                    results.append([1 - pred_soft_numpy[i][0]])
            results = np.array(results)

            opt_thres = OptThreshold()
            opt_result_txt = osp.join(self.train_dir, self.sample_dir,
                                      'test_opt_result_epoch_{}.txt'.format(str(epoch).zfill(3)))
            _, y_score_opt = opt_thres.find_optimal_cutoff(target_numpy, results, opt_result_txt)
            plotingConfusionMatrix(target_numpy, pred_idx_numpy, class_list, epoch,
                                   osp.join(self.train_dir, self.sample_dir), prefix='test_plain_')
            plotingConfusionMatrix(target_numpy, y_score_opt, class_list, epoch,
                                   osp.join(self.train_dir, self.sample_dir), prefix='test_opt_')
            calculateROC(target_numpy, results, 1, class_list, epoch, osp.join(self.train_dir, self.sample_dir),
                         prefix='test_')

            if self.use_tensorboard:
                self.logger.scalar_summary('test accuracy', test_accuracy, epoch)
                self.logger.scalar_summary('test_loss', testing_loss, epoch)

            self.G = self.G.train()

    def test(self):
        pass

