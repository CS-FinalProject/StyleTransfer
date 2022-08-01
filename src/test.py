import argparse
import os
import random

import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm import tqdm
from cyclegan.utils import Checkpoint
from train import load_checkpoint
from cyclegan.models.cycle_gan import CycleGAN
from cyclegan import ImageDataset
import wandb


def arguments_parsing():
    parser = argparse.ArgumentParser(
        description="Test model results of Style Transfer")
    parser.add_argument("--dataroot", type=str, default=".",
                        help="path to datasets. (default:./data)")
    parser.add_argument("--cuda", action="store_true", help="Enables cuda")
    parser.add_argument("--outf", default="./results",
                        help="folder to output images. (default: `./results`).")
    parser.add_argument("--image-size", type=int, default=256,
                        help="size of the data crop (squared assumed). (default:256)")
    parser.add_argument("--manualSeed", type=int,
                        help="Seed for initializing training. (default:none)")
    parser.add_argument("--model-path", type=str,
                        help="A path to a specific model, if not specified, the program will automatically load the latest one")

    args = parser.parse_args()

    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", args.manualSeed)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)

    return args


def init_folders(args) -> None:
    if not os.path.exists(args.outf):
        os.makedirs(args.outf)
    if not os.path.exists(os.path.join(args.outf, str(args.dataset), "A")):
        os.makedirs(os.path.join(args.outf, str(args.dataset), "A"))
    if not os.path.exists(os.path.join(args.outf, str(args.dataset), "B")):
        os.makedirs(os.path.join(args.outf, str(args.dataset), "B"))


def init_dataset(args) -> torch.utils.data.DataLoader:
    dataset = ImageDataset(root=os.path.join(args.dataroot, args.dataset),
                           transform=transforms.Compose([
                               transforms.Resize(args.image_size),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                           ]),
                           mode="test")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)
    return dataloader


def create_and_load_model(args, device, run) -> CycleGAN:
    model = CycleGAN(0.02, 0, True, device, 0)
    load_checkpoint(model, run)

    # Set model mode
    model.generator_A2B.eval()
    model.generator_B2A.eval()

    return model


def test(args, device, run):
    dataloader = init_dataset(args)
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))

    model = create_and_load_model(args, device, run)

    for i, data in progress_bar:
        # get batch size data
        real_images_A = data["A"].to(device)
        real_images_B = data["B"].to(device)

        # Generate output
        fake_image_A = 0.5 * (model(real_images_B, is_inverse=True).data + 1.0)
        fake_image_B = 0.5 * (model(real_images_A, is_inverse=False).data + 1.0)

        # Save image files
        vutils.save_image(fake_image_A.detach(), f"{args.outf}/{args.dataset}/A/{i + 1:04d}.png", normalize=True)
        vutils.save_image(fake_image_B.detach(), f"{args.outf}/{args.dataset}/B/{i + 1:04d}.png", normalize=True)

        progress_bar.set_description(f"Process images {i + 1} of {len(dataloader)}")


def main():
    run = wandb.init(project="style-transfer")
    args = arguments_parsing()
    device = torch.device("cuda:0" if args.cuda else "cpu")
    init_folders(args)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    test(args, device, run)
    wandb.finish()


if __name__ == "__main__":
    main()
