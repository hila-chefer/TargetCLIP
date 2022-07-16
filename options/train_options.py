from argparse import ArgumentParser
from configs.paths_config import model_paths


class TrainOptions:

    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        self.parser.add_argument('--exp_dir', type=str, help='Path to experiment output directory')
        self.parser.add_argument('--dataset_type', default='ffhq_encode', type=str,
                                 help='Type of dataset/experiment to run')
        self.parser.add_argument('--encoder_type', default='Encoder4Editing', type=str, help='Which encoder to use')

        self.parser.add_argument('--batch_size', default=4, type=int, help='Batch size for training')
        self.parser.add_argument('--test_batch_size', default=2, type=int, help='Batch size for testing and inference')
        self.parser.add_argument('--workers', default=4, type=int, help='Number of train dataloader workers')
        self.parser.add_argument('--test_workers', default=2, type=int,
                                 help='Number of test/inference dataloader workers')

        self.parser.add_argument('--learning_rate', default=0.0001, type=float, help='Optimizer learning rate')
        self.parser.add_argument('--optim_name', default='ranger', type=str, help='Which optimizer to use')
        self.parser.add_argument('--train_decoder', default=False, type=bool, help='Whether to train the decoder model')
        self.parser.add_argument('--start_from_latent_avg', action='store_true',
                                 help='Whether to add average latent vector to generate codes from encoder.')

        self.parser.add_argument('--stylegan_weights', default=model_paths['stylegan_ffhq'], type=str,
                                 help='Path to StyleGAN model weights')
        self.parser.add_argument('--stylegan_size', default=1024, type=int,
                                 help='size of pretrained StyleGAN Generator')
        self.parser.add_argument('--checkpoint_path', default='pretrained_models/e4e_ffhq_encode.pt', type=str, help='Path to pSp model checkpoint')

        self.parser.add_argument('--max_steps', default=500000, type=int, help='Maximum number of training steps')
        self.parser.add_argument('--num_consistency', default=5, type=int, help='Number of sources for consistency loss')
        self.parser.add_argument('--image_interval', default=100, type=int,
                                 help='Interval for logging train images during training')
        self.parser.add_argument('--board_interval', default=50, type=int,
                                 help='Interval for logging metrics to tensorboard')
        self.parser.add_argument('--val_interval', default=1000, type=int, help='Validation interval')
        self.parser.add_argument('--save_interval', default=None, type=int, help='Model checkpoint interval')

        # Progressive training
        self.parser.add_argument('--progressive_steps', nargs='+', type=int, default=None,
                                 help="The training steps of training new deltas. steps[i] starts the delta_i training")
        self.parser.add_argument('--progressive_start', type=int, default=None,
                                 help="The training step to start training the deltas, overrides progressive_steps")
        self.parser.add_argument('--progressive_step_every', type=int, default=2_000,
                                 help="Amount of training steps for each progressive step")

        # Save additional training info to enable future training continuation from produced checkpoints
        self.parser.add_argument('--save_training_data', action='store_true',
                                 help='Save intermediate training data to resume training from the checkpoint')
        self.parser.add_argument('--sub_exp_dir', default=None, type=str, help='Name of sub experiment directory')
        self.parser.add_argument('--keep_optimizer', action='store_true',
                                 help='Whether to continue from the checkpoint\'s optimizer')
        self.parser.add_argument('--resume_training_from_ckpt', default=None, type=str,
                                 help='Path to training checkpoint, works when --save_training_data was set to True')
        self.parser.add_argument('--update_param_list', nargs='+', type=str, default=None,
                                 help="Name of training parameters to update the loaded training checkpoint")

        # essence transfer specific
        self.parser.add_argument('--lambda_consistency', type=float, default=0.6, help="norm type of the deltas")
        self.parser.add_argument('--lambda_reg', type=float, default=3e-3, help="lambda for delta norm loss")
        self.parser.add_argument('--sources', type=str, default='pretrained_models/encoder_sources.pt',
                            help='path to sources to use for training')
    def parse(self):
        opts = self.parser.parse_args()
        return opts
