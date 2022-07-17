import numpy as np
import torch

import random

seed = 0

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
from experiments.common import *

def load_image(image_size, device, raw_image):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    image = transform(raw_image).unsqueeze(0).to(device)
    return image


def main(args):
    with torch.no_grad():
        device = 'cuda'

        id_loss = IDLoss().cuda().eval()

        transform = transforms.Compose([
            ToTensor(),
        ])

        args.latent = 512
        args.n_mlp = 8
        args.image_size = 224
        args.blip_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth'
        blip = blip_feature_extractor(pretrained=args.blip_url, image_size=args.image_size, vit='base')
        blip.eval()
        blip = blip.to(device)
        cosine = torch.nn.CosineSimilarity(dim=0)

        input_img_paths = sorted(glob.glob(os.path.join(args.input_img_path, '*.*')))
        style_img_paths = sorted(glob.glob(os.path.join(args.style_img_path, '*.*')))[:]
        num = 0

        id_losses = []
        cosine_losses_blip = []
        cosine_losses_clip = []
        target_id_losses = []

        num_sources = len(input_img_paths)

        for style_img_path in style_img_paths:
            print(num)
            print(f'reading style image {style_img_path}')

            curr_id_loss = 0
            curr_cosine_sim_loss_blip = 0
            curr_cosine_sim_loss_clip = 0
            curr_target_id_loss = 0

            name_style = os.path.splitext(os.path.basename(style_img_path))[0]
            img_style = cv2.imread(style_img_path, 1)
            img_style_ten = cv2ten(img_style, device)
            img_style = cv2.resize(img_style, (args.size, args.size))

            for input_img_path in input_img_paths:
                name_in = os.path.splitext(os.path.basename(input_img_path))[0]
                img_in = cv2.imread(input_img_path, 1)
                img_in_ten = cv2ten(img_in, device)
                img_in = cv2.resize(img_in, (args.size, args.size))

                # img_out = Image.open(f'{args.manipulations_path}/ref_{name_style}_input_{name_in}.png')
                img_out = Image.open(f'{args.manipulations_path}/{name_style}/{name_in}.png')
                img_out_ten = transform(img_out).unsqueeze(0).cuda()

                i_loss = id_loss(img_in_ten, img_out_ten)[0]
                curr_id_loss += i_loss.item()

                target_i_loss = id_loss(img_style_ten, img_out_ten)[0]
                curr_target_id_loss += target_i_loss.item()

                src_amp_features_blip, target_features_blip, src_amp_features_clip, target_features_clip = \
                    get_semantic_metrics(args, device, img_out_ten, img_style_ten, blip, source_amp_pil=img_out)

                # cosine similarity loss
                curr_cosine_sim_loss_blip += cosine(target_features_blip, src_amp_features_blip).item()
                curr_cosine_sim_loss_clip += cosine(target_features_clip, src_amp_features_clip).item()

            curr_id_loss = curr_id_loss / num_sources
            print(f"id loss in iteration {num} is {curr_id_loss}")
            id_losses.append(curr_id_loss)

            curr_target_id_loss = curr_target_id_loss / num_sources
            print(f"id loss for target in iteration {num} is {curr_target_id_loss}")
            target_id_losses.append(curr_target_id_loss)

            curr_cosine_sim_loss_blip = curr_cosine_sim_loss_blip / num_sources
            print(f"cosine blip loss in iteration {num} is {curr_cosine_sim_loss_blip}")
            cosine_losses_blip.append(curr_cosine_sim_loss_blip)

            curr_cosine_sim_loss_clip = curr_cosine_sim_loss_clip / num_sources
            print(f"cosine clip loss in iteration {num} is {curr_cosine_sim_loss_clip}")
            cosine_losses_clip.append(curr_cosine_sim_loss_clip)

            num += 1

        print('Done! id losses:', id_losses)
        print('target id losses:', target_id_losses)
        print('cosine losses blip:', cosine_losses_blip)
        print('cosine losses clip:', cosine_losses_clip)
        print(f'average id: {100 * (1- sum(id_losses) / len(id_losses))}')
        print(f'std id: {100 * np.std(id_losses)}')
        print(f'average target id: {100 * (1- sum(target_id_losses) / len(target_id_losses))}')
        print(f'std target id: {100 * np.std(target_id_losses)}')
        print(f'average cosine blip: {100 * sum(cosine_losses_blip) / len(cosine_losses_blip)}')
        print(f'std blip: {100 * np.std(cosine_losses_blip)}')
        print(f'average cosine clip: {100 * sum(cosine_losses_clip) / len(cosine_losses_clip)}')
        print(f'std clip: {100 * np.std(cosine_losses_clip)}')


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--style_img_path', type=str, default='target_celebs_aligned', help='path to style image')
    parser.add_argument('--input_img_path', type=str, default=None, help='path to source images', required=True)
    parser.add_argument('--manipulations_path', type=str, default='baseline/output', required=True,
                        help='path to images after the manipulation with the baseline')

    args = parser.parse_args()
    main(args)