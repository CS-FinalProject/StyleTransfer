import argparse
import os
import random
from tqdm import tqdm

import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
import wandb

from cyclegan.utils import GEN_A2B_PATH, GEN_B2A_PATH, DISC_A_PATH, DISC_B_PATH
from cyclegan import ImageDataset
from cyclegan.models.cycle_gan import CycleGAN

# W&B initialization
wandb.init(project="style-transfer", entity="haifa-uni-monet-gan")


def arguments_parsing():
    """
    Define arguments for the training process.
    """
    parser = argparse.ArgumentParser(
        description="PyTorch implements `Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial "
                    "Networks`")
    parser.add_argument("--dataroot", type=str, default=".",
                        help="path to datasets. (default:./data)")
    parser.add_argument("--epochs", default=200, type=int, metavar="N",
                        help="number of total epochs to run")
    parser.add_argument("--decay_epochs", type=int, default=100,
                        help="epoch to start linearly decaying the learning rate to 0. (default:100)")
    parser.add_argument("-b", "--batch-size", default=1, type=int,
                        metavar="N",
                        help="mini-batch size (default: 1), this is the total "
                             "batch size of all GPUs on the current node when "
                             "using Data Parallel or Distributed Data Parallel")
    parser.add_argument("--lr", type=float, default=0.0002,
                        help="learning rate. (default:0.0002)")
    parser.add_argument("--cuda", action="store_true", help="Enables cuda")
    parser.add_argument("--continue-training", type=bool, default=False,
                        help="If this flag is true, then the training will resume from the last checkpoint")
    parser.add_argument("--image-size", type=int, default=256,
                        help="size of the data crop (squared assumed). (default:256)")
    parser.add_argument("--outf", default="./outputs",
                        help="folder to output images. (default:`./outputs`).")
    parser.add_argument("--manualSeed", type=int,
                        help="Seed for initializing training. (default:none)")
    parser.add_argument("--save_model_freq", default=200, help="The program will save the model each N batches",
                        type=int)
    parser.add_argument("--lambda_param", default=10,
                        help="The lambda parameter introduced in the paper. It's a weight for the cycle loss",
                        type=float)

    args = parser.parse_args()

    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", args.manualSeed)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)

    return args


def init_weights_and_biases(args):
    wandb.init(project="style-transfer", entity="haifa-uni-monet-gan")
    wandb.config = {
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size
    }


def init_folders(args) -> None:
    """
    Initialize folders for saving the outputs.
    """
    # Make output folder for output images
    if not os.path.exists(args.outf):
        os.makedirs(args.outf)
    if not os.path.exists("models"):
        os.makedirs("models")
    if not os.path.exists(GEN_A2B_PATH):
        os.makedirs(GEN_A2B_PATH)
    if not os.path.exists(GEN_B2A_PATH):
        os.makedirs(GEN_B2A_PATH)
    if not os.path.exists(DISC_A_PATH):
        os.makedirs(DISC_A_PATH)
    if not os.path.exists(DISC_B_PATH):
        os.makedirs(DISC_B_PATH)
    if not os.path.exists(os.path.join(args.outf, "A")):
        os.makedirs(os.path.join(args.outf, "A"))
    if not os.path.exists(os.path.join(args.outf, "B")):
        os.makedirs(os.path.join(args.outf, "B"))


def init_models_counting(dir_path: str) -> int:
    """
    Gets the latest model count in the folder.
    :param dir_path: The path to all models of the model type.
    :return: The latest count
    """
    models = os.listdir(dir_path)
    models.sort()

    if len(models) == 0:
        return -1
    file = models[-1]
    if file.endswith(".pth"):
        return int(file.split(".")[0])
    else:
        return -1


def models_counting() -> dict:
    return {
        "genA2B": init_models_counting(GEN_A2B_PATH),
        "genB2A": init_models_counting(GEN_B2A_PATH),
        "discA": init_models_counting(DISC_A_PATH),
        "discB": init_models_counting(DISC_B_PATH),
    }


def init_dataset(args) -> torch.utils.data.DataLoader:
    dataset = ImageDataset(root=os.path.join(args.dataroot, "dataset"),
                           transform=transforms.Compose([
                               transforms.Resize(int(args.image_size * 1.12)),
                               transforms.RandomCrop(args.image_size),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                           mode="train",
                           unaligned=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    return dataloader


def models_checkpoints(real_imageA, real_imageB, args, epoch_idx: int, batch_idx: int, cycle_gan: CycleGAN):
    if batch_idx % args.save_model_freq != 0 or batch_idx == 0:
        return

    for model in cycle_gan.counters:
        cycle_gan.counters[model] += 1

    # Saving the current trained models
    torch.save(cycle_gan.generator_A2B.state_dict(),
               os.path.join(GEN_A2B_PATH, str(cycle_gan.counters["genA2B"]) + ".pth"))
    torch.save(cycle_gan.generator_B2A.state_dict(),
               os.path.join(GEN_B2A_PATH, str(cycle_gan.counters["genB2A"]) + ".pth"))
    torch.save(cycle_gan.discriminator_A.state_dict(),
               os.path.join(DISC_A_PATH, str(cycle_gan.counters["discA"]) + ".pth"))
    torch.save(cycle_gan.discriminator_B.state_dict(),
               os.path.join(DISC_B_PATH, str(cycle_gan.counters["discB"]) + ".pth"))

    # Upload to W&B

    wandb.save(os.path.join(GEN_A2B_PATH, str(cycle_gan.counters["genA2B"]) + ".pth"))
    wandb.save(os.path.join(GEN_B2A_PATH, str(cycle_gan.counters["genB2A"]) + ".pth"))
    wandb.save(os.path.join(DISC_A_PATH, str(cycle_gan.counters["discA"]) + ".pth"))
    wandb.save(os.path.join(DISC_B_PATH, str(cycle_gan.counters["discB"]) + ".pth"))

    # Save images generated by the latest models
    fake_image_B = cycle_gan.generator_A2B(real_imageA)
    fake_image_A = cycle_gan.generator_B2A(real_imageB)

    vutils.save_image(real_imageA,
                      os.path.join(args.outf, "A", "real_epoch_{}_batch_{}.png".format(epoch_idx, batch_idx)),
                      normalize=True)
    vutils.save_image(fake_image_B,
                      os.path.join(args.outf, "B", "fake_epoch_{}_batch_{}.png".format(epoch_idx, batch_idx)),
                      normalize=True)

    vutils.save_image(real_imageB,
                      os.path.join(args.outf, "B", "real_epoch_{}_batch_{}.png".format(epoch_idx, batch_idx)),
                      normalize=True)
    vutils.save_image(fake_image_A,
                      os.path.join(args.outf, "A", "fake_epoch_{}_batch_{}.png".format(epoch_idx, batch_idx)),
                      normalize=True)

    # Uploading to W&B
    real_imageA = wandb.Image(real_imageA, caption="real_epoch_{}_batch_{}".format(epoch_idx, batch_idx))
    fake_image_B = wandb.Image(fake_image_B, caption="fake_epoch_{}_batch_{}".format(epoch_idx, batch_idx))
    real_imageB = wandb.Image(real_imageB, caption="real_epoch_{}_batch_{}".format(epoch_idx, batch_idx))
    fake_image_A = wandb.Image(fake_image_A, caption="fake_epoch_{}_batch_{}".format(epoch_idx, batch_idx))

    wandb.log({
        "A Real Images": real_imageA,
        "B Fake Images": fake_image_B,
        "B Real Images": real_imageB,
        "A Fake Images": fake_image_A
    })


def train(args, device):
    dataloader = init_dataset(args)

    cycle_gan_model = CycleGAN(args.lr, args.lambda_param, args.continue_training, models_counting(), device)

    for epoch_idx in range(args.epochs):
        progress_bar = tqdm(dataloader, desc="Epoch {}".format(epoch_idx))
        for batch_idx, batch in enumerate(progress_bar):
            progress_bar.set_description(
                "Epoch ({}/{}) Batch ({}/{})".format(epoch_idx, args.epochs, batch_idx, len(dataloader)))

            image_A, image_B = batch["A"], batch["B"]
            image_A = image_A.to(device)
            image_B = image_B.to(device)

            losses = cycle_gan_model.train_model(image_A, image_B)
            wandb.log(losses)

            models_checkpoints(image_A, image_B, args, epoch_idx, batch_idx, cycle_gan_model)

        cycle_gan_model.step_lr_schedulers()


def main():
    args_parser = arguments_parsing()

    # Initialize a bunch of things
    init_weights_and_biases(args_parser)
    init_folders(args_parser)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not args_parser.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    device = torch.device("cuda:0" if args_parser.cuda and torch.cuda.is_available() else "cpu")

    train(args_parser, device)
    wandb.finish()


if __name__ == '__main__':
    main()
