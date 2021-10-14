import argparse
import os

import numpy
import torch
import torchvision
from torch import optim
from tqdm import tqdm
import clip
from criteria.clip_loss import CLIPLoss
from models.stylegan2.model import Generator
import math
import copy


def get_new_latent_code(g_ema):
    mean_latent = g_ema.mean_latent(4096)

    if args.latent_path:
        latent_code_init = torch.load(args.latent_path).cuda()
    elif args.mode == "edit":
        latent_code_init_not_trunc = torch.randn(1, 512).cuda()
        with torch.no_grad():
            _, latent_code_init,_ = g_ema([latent_code_init_not_trunc], return_latents=True,
                                        truncation=args.truncation, truncation_latent=mean_latent)
    else:
        latent_code_init = mean_latent.detach().clone().repeat(1, 18, 1)

    return latent_code_init


def get_latent(g_ema):
    mean_latent = g_ema.mean_latent(4096)
    latent_code_init_not_trunc = torch.randn(1, 512).cuda()
    with torch.no_grad():
        # _, latent_code_init = g_ema([latent_code_init_not_trunc], return_latents=True,
        #                             truncation=args.truncation, truncation_latent=mean_latent)
        _, latent_code_init,_ = g_ema([latent_code_init_not_trunc], return_latents=True,
                                    truncation=args.truncation, truncation_latent=mean_latent)

    direction = latent_code_init.detach().clone()
    direction.requires_grad = True
    return direction


def load_model():
    g_ema = Generator(args.stylegan_size, 512, 8)
    g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.cuda()
    return g_ema


def get_lr(t, initial_lr, rampdown=0.75, rampup=0.005):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def main(args):
        g_ema = load_model()

        dir_name = args.dir_name

        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        NUM_DIRECTIONS = 10
        NUM_IMAGES = args.num_images
        directions = [get_latent(g_ema) for _ in range(NUM_DIRECTIONS)]

        for j, latent in enumerate(directions):
            with torch.no_grad():
                dir, _ = g_ema([latent], input_is_latent=True, randomize_noise=False)
                torchvision.utils.save_image(dir, f"{dir_name}/dir_{j}.png", normalize=True, range=(-1, 1))

        latents = [None] * NUM_IMAGES
        latents_batches = []
        if not args.real_images:
            for k in range(args.num_batches):
                for n in range(NUM_IMAGES):
                    with torch.no_grad():
                        latents[n] = get_latent(g_ema)
                        latents[n].requires_grad = False
                latents_batches.append(copy.deepcopy(latents))
        else:
            data = torch.load(args.data_path)
            for k in range(args.num_batches):
                for n in range(NUM_IMAGES):
                    with torch.no_grad():
                        latents[n] = data[n].unsqueeze(0).cuda()
                        latents[n].requires_grad = False
            latents_batches = [latents]

        clip_loss = CLIPLoss(args)
        mse_loss = torch.nn.MSELoss()
        optimizer = optim.Adam(directions, lr=args.lr, weight_decay=args.weight_decay)
        directions_normalized = []

        import torchvision.transforms as transforms
        from PIL import Image
        transform = transforms.Compose([
            transforms.Resize((args.stylegan_size, args.stylegan_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        def tensor2im(var):
            var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
            var = ((var + 1) / 2)
            var[var < 0] = 0
            var[var > 1] = 1
            var = var * 255
            return Image.fromarray(var.astype('uint8'))

        if args.target_path is not None:
            tgt_lat = None
            # target is image from folder
            with torch.no_grad():
                img_target = Image.open(args.target_path)
                crop_size = min(img_target.size[0], img_target.size[1])
                crop = transforms.Compose([
                    transforms.CenterCrop(crop_size)])
                img_target = crop(img_target)
                img_target = transform(img_target).unsqueeze(0).cuda()
                torchvision.utils.save_image(img_target, f"{dir_name}/target.png",
                                             normalize=True, range=(-1, 1))
                print(img_target.shape)
                target_clip = clip_loss.encode(img_target)
                target_clip = target_clip / target_clip.norm(dim=-1)

        for dir_idx, direction in enumerate(directions):

            coefficients_batch = [[None] * NUM_IMAGES] * args.num_batches
            for k in range(args.num_batches):
                for n in range(NUM_IMAGES):
                    coefficient = torch.ones(1).to("cuda")
                    coefficient.requires_grad = True
                    coefficients_batch[k][n] = coefficient

            if args.target_path is None:
                # target is latent dir
                with torch.no_grad():
                    img_target, _ = g_ema([direction], input_is_latent=True, randomize_noise=False)
                    target_clip = clip_loss.encode(img_target)
                    target_clip = target_clip / target_clip.norm(dim=-1)
                    tgt_lat = torch.clone(direction)
                    tgt_lat = tgt_lat.detach()

            opt_loss = torch.Tensor([float("Inf")]).to("cuda")
            opt_dir = None
            opt_latents = [None] * NUM_IMAGES
            pbar = tqdm(range(args.step))

            for i in pbar:
                # calculate learning rate
                t = i / args.step
                lr = get_lr(t, args.lr)
                optimizer.param_groups[0]["lr"] = lr

                tot_loss = 0

                # import pdb; pdb.set_trace()
                for batch_idx, latents in enumerate(latents_batches):
                    # import pdb; pdb.set_trace()
                    coefficients = coefficients_batch[batch_idx]
                    optimizer_coeffs = optim.Adam(coefficients, lr=args.lr, weight_decay=0.01)
                    loss = torch.zeros(1).cuda()
                    diffs = [None] * NUM_IMAGES

                    # semantic err
                    target_semantic = torch.zeros(1).cuda()
                    similarities_loss = torch.zeros(1).cuda()
                    with torch.no_grad():
                        target_region = clip.tokenize("hair").cuda()
                        text_features = clip_loss.model.encode_text(target_region)
                        text_features /= text_features.norm(dim=-1, keepdim=True)

                    for j, latent in enumerate(latents):
                        coefficient = coefficients[j]
                        img_gen1, _ = g_ema([latent], input_is_latent=True, randomize_noise=False)
                        img_gen_amp, _ = g_ema([latent + direction * coefficient], input_is_latent=True,
                                               randomize_noise=False)

                        image_gen1_clip = clip_loss.encode(img_gen1)
                        image_gen_amp_clip = clip_loss.encode(img_gen_amp)
                        diff_clip = image_gen_amp_clip - image_gen1_clip
                        diff_clip = diff_clip / diff_clip.norm(dim=-1, keepdim=True)
                        diffs[j] = diff_clip

                        # semantic loss
                        image_gen_amp_clip_norm = image_gen_amp_clip / image_gen_amp_clip.norm(dim=-1)
                        similarity_gap = image_gen_amp_clip_norm @ target_clip.T
                        target_semantic += (1 - similarity_gap.view(-1)) / NUM_IMAGES

                    if args.lambda_consistency > 0:
                        matrix_amp = torch.stack(diffs).view(NUM_IMAGES, -1)
                        diffs_mat_amp = matrix_amp @ matrix_amp.T
                        ones_mat = torch.ones(diffs_mat_amp.shape[0]).cuda()
                        similarities_loss = torch.sum(ones_mat - diffs_mat_amp) / (NUM_IMAGES ** 2 - NUM_IMAGES)
                        loss += args.lambda_consistency * similarities_loss

                    # add semantic transfer loss
                    loss += args.lambda_transfer * target_semantic.reshape(loss.shape)

                    pbar.set_description(
                        (
                            f"loss: {loss.item():.4f}; consistency loss: {similarities_loss.view(-1).item():.4f};"
                            f"lr: {lr:.4f};"
                            f"transfer loss: {target_semantic.item():.4f}"
                        )
                    )

                    optimizer.zero_grad()
                    optimizer_coeffs.zero_grad()
                    loss.backward()
                    optimizer_coeffs.step()
                    optimizer.step()

                    tot_loss += loss

                with torch.no_grad():

                    if tot_loss < opt_loss:
                        numpy.save('{0}/direction{1}.npy'.format(args.dir_name, dir_idx),
                                   direction.detach().cpu().numpy())
                        opt_loss = tot_loss

                        if tgt_lat is not None:
                            img_gen_tgt, _ = g_ema([tgt_lat + direction], input_is_latent=True, randomize_noise=False)
                            torchvision.utils.save_image(img_gen_tgt, f"{dir_name}/img_gen_{dir_idx}_tgt.png",
                                                         normalize=True, range=(-1, 1))
                            img_gen_tgt, _ = g_ema([tgt_lat - direction], input_is_latent=True, randomize_noise=False)
                            torchvision.utils.save_image(img_gen_tgt, f"{dir_name}/img_gen_{dir_idx}_tgt_red.png",
                                                         normalize=True, range=(-1, 1))

                        for batch_idx, latents in enumerate(latents_batches):
                            coefficients = coefficients_batch[batch_idx]
                            for j, latent in enumerate(latents):
                                coefficient = coefficients[j]
                                img_gen1, _ = g_ema([latent], input_is_latent=True, randomize_noise=False)
                                img_gen_amp, _ = g_ema([latent + direction * coefficient], input_is_latent=True,
                                                       randomize_noise=False)

                                torchvision.utils.save_image(img_gen1, f"{dir_name}/img_gen_b_{batch_idx}_{j}.png",
                                                             normalize=True, range=(-1, 1))
                                torchvision.utils.save_image(img_gen_amp,
                                                             f"{dir_name}/img_gen_amp_{dir_idx}_b_{batch_idx}_{j}.png",
                                                             normalize=True, range=(-1, 1))


if __name__ == "__main__":
    torch.manual_seed(12345)
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_consistency", type=float, default=0.6)
    parser.add_argument("--dir_name", type=str, default="results_transfer",
                        help="name of directory to store results")
    parser.add_argument("--ckpt", type=str, default="./pretrained_models/stylegan2-ffhq-config-f.pt",
                        help="pretrained StyleGAN2 weights")
    parser.add_argument("--stylegan_size", type=int, default=1024, help="StyleGAN resolution")
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--weight_decay", type=float, default=3e-3)
    parser.add_argument("--step", type=int, default=1000, help="number of optimization steps")
    parser.add_argument("--target_path", type=str, default=None,
                        help="starts the optimization from the given latent code if provided. Otherwose, starts from"
                             "the mean latent in a free generation, and from a random one in editing. "
                             "Expects a .pt format")
    parser.add_argument("--truncation", type=float, default=0.7,
                        help="used only for the initial latent vector, and only when a latent code path is"
                             "not provided")
    parser.add_argument("--save_intermediate_image_every", type=int, default=1,  # 150,
                        help="if > 0 then saves intermidate results during the optimization")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--lambda_transfer", type=float, default=1)
    parser.add_argument("--lambda_reg", type=float, default=0)
    parser.add_argument("--num_images", type=int, default=6, help="StyleGAN resolution")
    parser.add_argument("--num_batches", type=int, default=1, help="StyleGAN resolution")
    parser.add_argument("--real_images", default=True, action='store_true')
    parser.add_argument("--data_path", type=str, default="train_faces.pt")

    args = parser.parse_args()

    result_image = main(args)