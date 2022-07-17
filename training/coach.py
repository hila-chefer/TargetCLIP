import os
import random
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import common, train_utils
from configs import data_configs
from datasets.images_dataset import ImagesDataset
from models.psp import pSp
from models.encoders.psp_encoders import ProgressiveStage
from training.ranger import Ranger
from criteria.clip_loss import CLIPLoss
import torchvision.transforms as transforms
random.seed(0)
torch.manual_seed(0)

class Essence:
    def __init__(self, generator, opts):
        """
        Latent Editing Consistency metric as proposed in the main paper.
        :param generator: the stylegan generator to use
        :param num_consistency: number of consistency samples to account for
        """
        self.opts = opts
        self.stylegan_size = generator.size
        self.g_ema = self.load_model()
        self.sources = torch.load(opts.sources).to('cuda:1')
        self.num_consistency = opts.num_consistency
        self.num_sources = self.sources.shape[0]
        clip_loss = CLIPLoss(self.stylegan_size)
        self.clip_loss = torch.nn.DataParallel(clip_loss, device_ids=[1, 2, 3, 4, 0])
        self.regularization_loss = torch.nn.MSELoss(reduction='sum')

    def load_model(self):
        from models.stylegan2.model import Generator
        g_ema = Generator(self.opts.stylegan_size, 512, 8)
        g_ema.load_state_dict(torch.load(self.opts.stylegan_weights)["g_ema"], strict=False)
        g_ema.eval()
        g_ema = torch.nn.DataParallel(g_ema, device_ids=[1, 2, 3, 4, 0])
        g_ema = g_ema.to('cuda:1')
        return g_ema

    def forward(self, essence, target):
        latents = self.sources

        # encode target image
        transform = transforms.Compose([
            transforms.Resize((self.opts.stylegan_size, self.opts.stylegan_size)),
        ])
        target = transform(target)
        batch_size = target.shape[0]
        similarity_loss = 0
        consistency_loss = 0
        l2_loss = 0
        for i in range(batch_size):
            curr_essence = essence[i]
            target_clip = self.clip_loss.module.encode(target[i].unsqueeze(0))
            target_clip = target_clip / target_clip.norm(dim=-1)

            with torch.no_grad():
                img_gen, _ = self.g_ema([latents], input_is_latent=True, randomize_noise=False)
                image_gen_clip = self.clip_loss.module.encode(img_gen)

            direction_with_coeff = [curr_essence for i in range(self.num_consistency)]
            direction_with_coeff = torch.stack(direction_with_coeff).squeeze(1).to('cuda:1')
            img_gen_amp, _ = self.g_ema([latents + direction_with_coeff], input_is_latent=True, randomize_noise=False)

            image_gen_amp_clip = self.clip_loss.module.encode(img_gen_amp)
            diffs = image_gen_amp_clip - image_gen_clip
            diffs = diffs / diffs.norm(dim=-1, keepdim=True)

            # similarity loss
            image_gen_amp_clip_norm = image_gen_amp_clip / image_gen_amp_clip.norm(dim=-1, keepdim=True)
            similarity_gap = image_gen_amp_clip_norm @ target_clip.T
            similarity_loss += 1 - similarity_gap.mean()

            # consistency loss
            diffs_mat_amp = diffs @ diffs.T
            ones_mat = torch.ones(diffs_mat_amp.shape[0]).to(diffs_mat_amp.device)
            similarities_loss = torch.sum(ones_mat - diffs_mat_amp) / (self.num_consistency ** 2 - self.num_consistency)
            consistency_loss += similarities_loss

            # regularization
            l2_loss += torch.norm(essence)

        similarity_loss /= batch_size
        consistency_loss /= batch_size
        l2_loss /= batch_size

        return similarity_loss, consistency_loss, l2_loss


class Coach:
    def __init__(self, opts, prev_train_checkpoint=None):
        self.opts = opts

        self.global_step = 0

        self.device = 'cuda:0'
        self.opts.device = self.device
        # Initialize network
        self.net = pSp(self.opts).to(self.device)

        # Initialize optimizer
        self.optimizer = self.configure_optimizers()

        # Initialize dataset
        self.train_dataset, self.test_dataset = self.configure_datasets()
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=self.opts.batch_size,
                                           shuffle=True,
                                           num_workers=int(self.opts.workers),
                                           drop_last=True)
        self.test_dataloader = DataLoader(self.test_dataset,
                                          batch_size=self.opts.test_batch_size,
                                          shuffle=False,
                                          num_workers=int(self.opts.test_workers),
                                          drop_last=True)

        # Initialize logger
        log_dir = os.path.join(opts.exp_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.logger = SummaryWriter(log_dir=log_dir)

        # Initialize checkpoint dir
        self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_val_loss = None
        if self.opts.save_interval is None:
            self.opts.save_interval = self.opts.max_steps

        if prev_train_checkpoint is not None:
            self.load_from_train_checkpoint(prev_train_checkpoint)
            prev_train_checkpoint = None

        # essence transfer
        self.essence_loss = Essence(self.net.decoder, opts)

    def load_from_train_checkpoint(self, ckpt):
        print('Loading previous training data...')
        self.global_step = ckpt['global_step'] + 1
        self.best_val_loss = ckpt['best_val_loss']
        self.net.load_state_dict(ckpt['state_dict'])

        if self.opts.keep_optimizer:
            self.optimizer.load_state_dict(ckpt['optimizer'])
        if self.opts.progressive_steps:
            self.check_for_progressive_training_update(is_resume_from_ckpt=True)
        print(f'Resuming training from step {self.global_step}')

    def train(self):
        self.net.train()
        if self.opts.progressive_steps:
            self.check_for_progressive_training_update()
        while self.global_step < self.opts.max_steps:
            for batch_idx, batch in enumerate(self.train_dataloader):
                loss_dict = {}
                x, y, y_hat, latent = self.forward(batch)
                loss, encoder_loss_dict, id_logs = self.calc_loss(x, y, y_hat, latent)
                loss_dict = {**loss_dict, **encoder_loss_dict}
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Logging related
                if self.global_step % self.opts.image_interval == 0 or (
                        self.global_step < 1000 and self.global_step % 25 == 0):
                    self.parse_and_log_images(x, self.opts, latent, title='images/train/faces')
                if self.global_step % self.opts.board_interval == 0:
                    self.print_metrics(loss_dict, prefix='train')
                    self.log_metrics(loss_dict, prefix='train')

                # Validation related
                val_loss_dict = None
                if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
                    val_loss_dict = self.validate()
                    if val_loss_dict and (self.best_val_loss is None or val_loss_dict['tot_loss'] < self.best_val_loss):
                        self.best_val_loss = val_loss_dict['tot_loss']
                        self.checkpoint_me(val_loss_dict, is_best=True)

                if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
                    if val_loss_dict is not None:
                        self.checkpoint_me(val_loss_dict, is_best=False)
                    else:
                        self.checkpoint_me(loss_dict, is_best=False)

                if self.global_step == self.opts.max_steps:
                    print('OMG, finished training!')
                    break

                self.global_step += 1
                if self.opts.progressive_steps:
                    self.check_for_progressive_training_update()

    def check_for_progressive_training_update(self, is_resume_from_ckpt=False):
        for i in range(len(self.opts.progressive_steps)):
            if is_resume_from_ckpt and self.global_step >= self.opts.progressive_steps[i]:  # Case checkpoint
                self.net.encoder.set_progressive_stage(ProgressiveStage(i))
            if self.global_step == self.opts.progressive_steps[i]:   # Case training reached progressive step
                self.net.encoder.set_progressive_stage(ProgressiveStage(i))

    def validate(self):
        self.net.eval()
        agg_loss_dict = []

        for batch_idx, batch in enumerate(self.test_dataloader):
            cur_loss_dict = {}
            with torch.no_grad():
                x, y, y_hat, latent = self.forward(batch)
                loss, cur_encoder_loss_dict, id_logs = self.calc_loss(x, y, y_hat, latent)
                cur_loss_dict = {**cur_loss_dict, **cur_encoder_loss_dict}
            agg_loss_dict.append(cur_loss_dict)

            # Logging related
            self.parse_and_log_images(x, self.opts, latent,
                                      title='images/test/faces',
                                      subscript='{:04d}'.format(batch_idx))


            # only process 50 eval images
            if batch_idx >= 50:
                break

        loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
        self.log_metrics(loss_dict, prefix='test')
        self.print_metrics(loss_dict, prefix='test')

        self.net.train()
        return loss_dict

    def checkpoint_me(self, loss_dict, is_best):
        save_name = 'best_model.pt' if is_best else 'iteration_{}.pt'.format(self.global_step)
        save_dict = self.__get_save_dict()
        checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
        torch.save(save_dict, checkpoint_path)
        with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
            if is_best:
                f.write(
                    '**Best**: Step - {}, Loss - {:.3f} \n{}\n'.format(self.global_step, self.best_val_loss, loss_dict))
            else:
                f.write('Step - {}, \n{}\n'.format(self.global_step, loss_dict))

    def configure_optimizers(self):
        params = list(self.net.encoder.parameters())
        if self.opts.train_decoder:
            params += list(self.net.decoder.parameters())
        else:
            self.requires_grad(self.net.decoder, False)
        if self.opts.optim_name == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
        else:
            optimizer = Ranger(params, lr=self.opts.learning_rate)
        return optimizer

    def configure_datasets(self):
        if self.opts.dataset_type not in data_configs.DATASETS.keys():
            Exception('{} is not a valid dataset_type'.format(self.opts.dataset_type))
        print('Loading dataset for {}'.format(self.opts.dataset_type))
        dataset_args = data_configs.DATASETS[self.opts.dataset_type]
        transforms_dict = dataset_args['transforms'](self.opts).get_transforms()
        train_dataset = ImagesDataset(source_root=dataset_args['train_source_root'],
                                      target_root=dataset_args['train_target_root'],
                                      source_transform=transforms_dict['transform_source'],
                                      target_transform=transforms_dict['transform_gt_train'],
                                      opts=self.opts)
        test_dataset = ImagesDataset(source_root=dataset_args['test_source_root'],
                                     target_root=dataset_args['test_target_root'],
                                     source_transform=transforms_dict['transform_source'],
                                     target_transform=transforms_dict['transform_test'],
                                     opts=self.opts)
        print("Number of training samples: {}".format(len(train_dataset)))
        print("Number of test samples: {}".format(len(test_dataset)))
        return train_dataset, test_dataset


    def calc_loss(self, x, y, y_hat, latent):
        loss_dict = {}
        loss = 0.0
        id_logs = None
        similarity_loss, consistency_loss, l2_loss = self.essence_loss.forward(latent, x)
        loss_dict['similarity_loss'] = similarity_loss
        loss_dict['consistency_loss'] = consistency_loss
        loss_dict['l2_loss'] = l2_loss
        loss = similarity_loss + self.opts.lambda_consistency * consistency_loss + self.opts.lambda_reg * l2_loss
        loss_dict['l2_norm'] = latent.norm()
        loss_dict['tot_loss'] = loss
        return loss, loss_dict, id_logs

    def forward(self, batch):
        x, y = batch
        x, y = x.to(self.device).float(), y.to(self.device).float()
        y_hat, latent = self.net.forward(x, return_latents=True)
        if self.opts.dataset_type == "cars_encode":
            y_hat = y_hat[:, :, 32:224, :]
        return x, y, y_hat, latent

    def log_metrics(self, metrics_dict, prefix):
        for key, value in metrics_dict.items():
            self.logger.add_scalar('{}/{}'.format(prefix, key), value, self.global_step)

    def print_metrics(self, metrics_dict, prefix):
        print('Metrics for {}, step {}'.format(prefix, self.global_step))
        for key, value in metrics_dict.items():
            print('\t{} = '.format(key), value)

    def parse_and_log_images(self, x, opts, latent, title, subscript=None, display_count=5):
        im_data = []
        latents = torch.load(opts.sources).cuda()
        for i in range(display_count):
            img_gen, _ = self.net.decoder([latents[i].unsqueeze(0)], input_is_latent=True, randomize_noise=False)
            img_gen_amp, _ = self.net.decoder([latents[i].unsqueeze(0) + latent], input_is_latent=True, randomize_noise=False)
            cur_im_data = {
                'input_face': common.log_input_image(img_gen[0], self.opts),
                'target_face': common.log_input_image(x[0], self.opts),
                'output_face': common.tensor2im(img_gen_amp[0]),
            }
            im_data.append(cur_im_data)
        self.log_images(title, im_data=im_data, subscript=subscript)

    def log_images(self, name, im_data, subscript=None, log_latest=False):
        fig = common.vis_faces(im_data)
        step = self.global_step
        if log_latest:
            step = 0
        if subscript:
            path = os.path.join(self.logger.log_dir, name, '{}_{:04d}.jpg'.format(subscript, step))
        else:
            path = os.path.join(self.logger.log_dir, name, '{:04d}.jpg'.format(step))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)

    def __get_save_dict(self):
        save_dict = {
            'state_dict': self.net.state_dict(),
            'opts': vars(self.opts)
        }
        # save the latent avg in state_dict for inference if truncation of w was used during training
        if self.opts.start_from_latent_avg:
            save_dict['latent_avg'] = self.net.latent_avg

        if self.opts.save_training_data:  # Save necessary information to enable training continuation from checkpoint
            save_dict['global_step'] = self.global_step
            save_dict['optimizer'] = self.optimizer.state_dict()
            save_dict['best_val_loss'] = self.best_val_loss
        return save_dict


    def is_progressive_training(self):
        return self.opts.progressive_steps is not None

    @staticmethod
    def requires_grad(model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag