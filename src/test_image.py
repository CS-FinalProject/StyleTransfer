import argparse
import os.path
import random
import timeit

import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.utils as vutils
import wandb
from PIL import Image

from cyclegan.models.cycle_gan import CycleGAN
from train import load_checkpoint

parser = argparse.ArgumentParser(
    description="Style Transfer")
parser.add_argument("--file", type=str,
                    help="Image name. (default:`assets/horse.png`)")
parser.add_argument("--model-path", type=str,
                    help="Path to a model file, if wanted from a specific model",
                    default=os.path.join("..", "final_model.pth"))
parser.add_argument("--cuda", action="store_true", help="Enables cuda")
parser.add_argument("--image-size", type=int, default=256,
                    help="size of the scripts crop (squared assumed). (default:256)")
parser.add_argument("--manualSeed", type=int,
                    help="Seed for initializing training. (default:none)")
parser.add_argument("--is-inverse", type=bool, help="Map from B to A", default=True)
parser.add_argument("--outf", type=str, help="Path of result image", default=False)

args = parser.parse_args()
print(args)

# run = wandb.init(project="style-transfer")

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda:0" if args.cuda else "cpu")

# create model
model = CycleGAN(0, 0, True, device, 0, 0)

# load_checkpoint(model, run, device)
checkpoint = torch.load(args.model_path, map_location=device)
model.generator_A2B.load_state_dict(checkpoint.genA2B)
model.generator_B2A.load_state_dict(checkpoint.genB2A)

model = model.to(device)

# Set model mode
# model.eval()

# Load image
image = Image.open(args.file)
pre_process = transforms.Compose([transforms.Resize(args.image_size),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                                  ])
image = pre_process(image).unsqueeze(0)
image = image.to(device)

start = timeit.default_timer()
fake_image = model(image, args.is_inverse)
elapsed = (timeit.default_timer() - start)
print(f"cost {elapsed:.4f}s")
vutils.save_image(fake_image.detach(), os.path.join(args.outf, "result.png"), normalize=True)

