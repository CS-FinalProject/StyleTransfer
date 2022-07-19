import argparse
import os
import random
import wandb

import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from cyclegan import DecayLR
from cyclegan import Discriminator
from cyclegan import Generator
from cyclegan import ImageDataset
from cyclegan import ReplayBuffer
from cyclegan import weights_init
from cyclegan import GEN_A2B_PATH, GEN_B2A_PATH, DISC_A_PATH, DISC_B_PATH

# W&B initialization
wandb.init(project="style-transfer", entity="haifa-uni-monet-gan")


def arguments_parsing():
    """
    Define arguments for the training process.
    """
    parser = argparse.ArgumentParser(
        description="PyTorch implements `Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial "
                    "Networks`")
    parser.add_argument("--dataroot", type=str, default="..",
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
    parser.add_argument("--save_model_freq", default=20, help="The program will save the model each N batches",
                        type=int)

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
    if not os.path.exists(os.path.join("outputs", "A")):
        os.makedirs(os.path.join("outputs", "A"))
    if not os.path.exists(os.path.join("outputs", "B")):
        os.makedirs(os.path.join("outputs", "B"))


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


def init_models(args, device, counters: dict) -> tuple:
    generator_A2B = Generator().to(device)
    generator_B2A = Generator().to(device)
    discriminator_A = Discriminator().to(device)
    discriminator_B = Discriminator().to(device)

    # Load the latest models if we want to continue learning
    if args.continue_training and len(os.listdir(GEN_A2B_PATH)) > 0:
        generator_A2B.load_state_dict(torch.load(os.path.join(GEN_A2B_PATH, str(counters["genA2B"]) + ".pth")))
        generator_B2A.load_state_dict(torch.load(os.path.join(GEN_B2A_PATH, str(counters["genB2A"]) + ".pth")))
        discriminator_A.load_state_dict(torch.load(os.path.join(DISC_A_PATH, str(counters["discA"]) + ".pth")))
        discriminator_B.load_state_dict(torch.load(os.path.join(DISC_B_PATH, str(counters["discB"]) + ".pth")))
    else:
        generator_A2B.apply(weights_init)
        generator_B2A.apply(weights_init)
        discriminator_A.apply(weights_init)
        discriminator_B.apply(weights_init)

    generator_A2B.train()
    generator_B2A.train()
    discriminator_A.train()
    discriminator_B.train()

    return generator_A2B, generator_B2A, discriminator_A, discriminator_B


def models_checkpoints(real_imageA, real_imageB, args, epoch_idx: int, batch_idx: int, generator_A2B: Generator,
                       generator_B2A: Generator,
                       discriminator_A: Discriminator, discriminator_B: Discriminator, counters: dict):
    if batch_idx % args.save_model_freq != 0:
        return

    for model in counters:
        counters[model] += 1

    # Saving the current trained models
    torch.save(generator_A2B.state_dict(), os.path.join(GEN_A2B_PATH, str(counters["genA2B"]) + ".pth"))
    torch.save(generator_B2A.state_dict(), os.path.join(GEN_B2A_PATH, str(counters["genB2A"]) + ".pth"))
    torch.save(discriminator_A.state_dict(), os.path.join(DISC_A_PATH, str(counters["discA"]) + ".pth"))
    torch.save(discriminator_B.state_dict(), os.path.join(DISC_B_PATH, str(counters["discB"]) + ".pth"))

    # Save images generated by the latest models
    fake_image_B = generator_A2B(real_imageA)
    fake_image_A = generator_B2A(real_imageB)

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
    wandb.log({"A Real Images": real_imageA})
    fake_image_B = wandb.Image(fake_image_B, caption="fake_epoch_{}_batch_{}".format(epoch_idx, batch_idx))
    wandb.log({"B Fake Images": fake_image_B})

    real_imageB = wandb.Image(real_imageB, caption="real_epoch_{}_batch_{}".format(epoch_idx, batch_idx))
    wandb.log({"B Real Images": real_imageB})
    fake_image_A = wandb.Image(fake_image_A, caption="fake_epoch_{}_batch_{}".format(epoch_idx, batch_idx))
    wandb.log({"A Fake Images": fake_image_A})


def train(args, device):
    dataloader = init_dataset(args)

    counters = {
        "genA2B": init_models_counting(GEN_A2B_PATH),
        "genB2A": init_models_counting(GEN_B2A_PATH),
        "discA": init_models_counting(DISC_A_PATH),
        "discB": init_models_counting(DISC_B_PATH),
    }

    # Fetch all generators and discriminators
    generator_A2B, generator_B2A, discriminator_A, discriminator_B = init_models(args, device, counters)

    # Define loss functions
    loss_func_adversarial = nn.MSELoss()
    loss_func_cycle_gan = nn.MSELoss()
    loss_func_identity_loss = nn.L1Loss()

    # Define optimizers
    genA2B_optim = torch.optim.Adam(generator_A2B.parameters(), lr=args.lr)
    genB2A_optim = torch.optim.Adam(generator_B2A.parameters(), lr=args.lr)
    discA_optim = torch.optim.Adam(discriminator_A.parameters(), lr=args.lr)
    discB_optim = torch.optim.Adam(discriminator_B.parameters(), lr=args.lr)
    optimizers = [genA2B_optim, genB2A_optim, discA_optim, discB_optim]

    # Define losses
    genB2A_losses = {"cycle_loss": [], "adversarial_loss": []}
    genA2B_losses = {"cycle_loss": [], "adversarial_loss": []}
    discA_losses = {"identity_loss": []}
    discB_losses = {"identity_loss": []}

    lr_lambda = DecayLR(args.epochs, 0, args.decay_epochs).step
    lr_scheduler_genA2B = torch.optim.lr_scheduler.LambdaLR(genA2B_optim, lr_lambda=lr_lambda)
    lr_scheduler_genB2A = torch.optim.lr_scheduler.LambdaLR(genB2A_optim, lr_lambda=lr_lambda)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(discA_optim, lr_lambda=lr_lambda)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(discB_optim, lr_lambda=lr_lambda)

    def adversarial_loss(x, y, discriminator: Discriminator, generator: Generator, losses: dict):
        adv_loss = loss_func_adversarial(discriminator(x), discriminator(generator(y)))
        losses["adversarial_loss"].append(adv_loss.item())
        return adv_loss

    def cycle_loss(x, y, generator: Generator, inverse_generator: Generator, losses: dict):
        cyc_loss = loss_func_cycle_gan(inverse_generator(generator(x)), x) + \
                   loss_func_cycle_gan(generator(inverse_generator(y)), y)

        losses["cycle_loss"].append(cyc_loss.item())
        return cyc_loss

    def update_discriminator(discriminator: Discriminator, generator: Generator, disc_optimizer, image, losses: dict):
        generator.eval()
        discriminator.train()

        disc_optimizer.zero_grad()
        disc_output = discriminator(image)
        err = loss_func_identity_loss(disc_output, torch.full(disc_output.size(), 1, device=device))
        err.backward()

        inverse_err = loss_func_identity_loss(discriminator(generator(image)),
                                              torch.full(disc_output.size(), 0, device=device))
        inverse_err.backward()

        losses["identity_loss"].append(err.item() + inverse_err.item())
        disc_optimizer.step()

        generator.train()
        return err + inverse_err

    for epoch_idx in range(args.epochs):
        progress_bar = tqdm(dataloader, desc="Epoch {}".format(epoch_idx))
        for batch_idx, batch in enumerate(progress_bar):
            progress_bar.set_description(
                "Epoch ({}/{}) Batch ({}/{})".format(epoch_idx, args.epochs, batch_idx, len(dataloader)))
            image_A, image_B = batch["A"], batch["B"]
            image_A = image_A.to(device)
            image_B = image_B.to(device)

            for optimizer in optimizers:
                optimizer.zero_grad()

            #########################################################
            # Update the generators with CycleGAN and Adversarial   #
            #########################################################

            # We don't want to update the discriminators' weights during the generators update
            discriminator_A.eval()
            discriminator_B.eval()

            # Adversarial Loss
            adversarial_loss_A = adversarial_loss(image_A, image_B, discriminator_A, generator_B2A, genB2A_losses)
            adversarial_loss_B = adversarial_loss(image_B, image_A, discriminator_B, generator_A2B, genA2B_losses)

            # Cycle GAN loss
            cycle_loss_A = cycle_loss(image_A, image_B, generator_A2B, generator_B2A, genA2B_losses)  # A -> B
            cycle_loss_B = cycle_loss(image_B, image_A, generator_B2A, generator_A2B, genB2A_losses)  # B -> A

            total_error = adversarial_loss_A + adversarial_loss_B + cycle_loss_A + cycle_loss_B
            total_error.backward()

            genA2B_optim.step()
            genB2A_optim.step()

            # Log in W&B
            wandb.log({"Generator A2B Loss": cycle_loss_A + adversarial_loss_B,
                       "Generator B2A Loss": cycle_loss_B + adversarial_loss_A})

            #####################################################
            # Update the discriminators using Identity loss     #
            #####################################################

            # Discriminator A
            discA_loss = update_discriminator(discriminator_A, generator_B2A, discA_optim, image_A, discA_losses)
            # Discriminator B
            discB_loss = update_discriminator(discriminator_B, generator_A2B, discB_optim, image_B, discB_losses)

            wandb.log({"Discriminator A Loss": discA_loss,
                       "Discriminator B Loss": discB_loss})

            #####################################################
            # Update the learning rates                          #
            #####################################################

            lr_scheduler_genA2B.step()
            lr_scheduler_genB2A.step()
            lr_scheduler_D_A.step()
            lr_scheduler_D_B.step()

            # Handle saving of models and output images
            models_checkpoints(image_A, image_B, args, epoch_idx, batch_idx, generator_A2B, generator_B2A,
                               discriminator_A, discriminator_B, counters)

            # Update learning rates
            lr_scheduler_genA2B.step()
            lr_scheduler_genB2A.step()
            lr_scheduler_D_A.step()
            lr_scheduler_D_B.step()


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
