import argparse
import os
import random

import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm import tqdm

from cyclegan_pytorch import Generator
from cyclegan_pytorch import ImageDataset

def arguments_parsing():
    parser = argparse.ArgumentParser(
        description="PyTorch implements `Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks`")
    parser.add_argument("--dataroot", type=str, default="./data",
                        help="path to datasets. (default:./data)")
    parser.add_argument("--dataset", type=str, default="horse2zebra",
                        help="dataset name. (default:`horse2zebra`)"
                             "Option: [apple2orange, summer2winter_yosemite, horse2zebra, monet2photo, "
                             "cezanne2photo, ukiyoe2photo, vangogh2photo, maps, facades, selfie2anime, "
                             "iphone2dslr_flower, ae_photos, ]")
    parser.add_argument("--cuda", action="store_true", help="Enables cuda")
    parser.add_argument("--outf", default="./results",
                        help="folder to output images. (default: `./results`).")
    parser.add_argument("--image-size", type=int, default=256,
                        help="size of the data crop (squared assumed). (default:256)")
    parser.add_argument("--manualSeed", type=int,
                        help="Seed for initializing training. (default:none)")

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


def create_and_load_model(args, device) -> tuple:
    # create models
    generator_A2B = Generator().to(device)
    generator_B2A = Generator().to(device)

    # Load state dicts
    generator_A2B.load_state_dict(torch.load(os.path.join("weights", str(args.dataset), "netG_A2B.pth")))
    generator_B2A.load_state_dict(torch.load(os.path.join("weights", str(args.dataset), "netG_B2A.pth")))

    # Set model mode
    generator_A2B.eval()
    generator_B2A.eval()

    return generator_A2B, generator_B2A


def test(args, device):
    dataloader = init_dataset(args)
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))

    for i, data in progress_bar:
        # get batch size data
        real_images_A = data["A"].to(device)
        real_images_B = data["B"].to(device)

        # Generate output
        generator_A2B, generator_B2A = creat_and_load_model(args, device)
        fake_image_A = 0.5 * (generator_B2A(real_images_B).data + 1.0)
        fake_image_B = 0.5 * (generator_A2B(real_images_A).data + 1.0)

        # Save image files
        vutils.save_image(fake_image_A.detach(), f"{args.outf}/{args.dataset}/A/{i + 1:04d}.png", normalize=True)
        vutils.save_image(fake_image_B.detach(), f"{args.outf}/{args.dataset}/B/{i + 1:04d}.png", normalize=True)

        progress_bar.set_description(f"Process images {i + 1} of {len(dataloader)}")


def main():
    args = arguments_parsing()
    device = torch.device("cuda:0" if args.cuda else "cpu")
    init_folders(args)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    test(device)


if __name__ == "__main__":
    main()