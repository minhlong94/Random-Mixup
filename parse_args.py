import argparse
import time


def parse():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(
        description="Train classifier with mixup, controlled by a RL agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    id = int(time.time())

    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar100",
        choices=["cifar10", "cifar100", "imagenet"],
        help="Dataset.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="~/Datasets/",
        help="path to download train dataset (CIFAR) or load dataset (ImageNet/TinyImageNet).",
    )
    parser.add_argument(
        "--val_data_dir",
        type=str,
        default="~/Datasets/",
        help="path to download val dataset (CIFAR) or load dataset (ImageNet/TinyImageNet).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="preactresnet18",
        choices=[
            "preactresnet18",
            "resnext29_4_24",
            "wrn16_8",
            "resnet50",
            "wrn28_10",
        ],
    )
    parser.add_argument(
        "--dirichlet_alpha",
        type=float,
        default=1.0,
        help="dirichlet alpha for weight sampling",
    )
    parser.add_argument(
        "--num_patches",
        type=int,
        default=8,
        help="Number of patches to divide the image. "
        "image will be sliced to have num_patches x num_patches patches",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.2,
        help="initial learning rate. Will be overwritten if use OneCycleLR scheduler",
    )
    parser.add_argument(
        "--seed", type=int, default=id, help="global seed. Default to int(time.time())"
    )

    parser.add_argument("--use_wandb", type=str2bool, default=False, help="use wandb")
    parser.add_argument(
        "--wandb_offline",
        type=str2bool,
        default=True,
        help="whether to use offline wandb",
    )
    parser.add_argument("--wandb_key", type=str, default="", help="wandb key")
    parser.add_argument(
        "--wandb_project", type=str, default="", help="wandb project name"
    )
    parser.add_argument("--wandb_entity", type=str, default="", help="wandb entity")
    parser.add_argument("--wandb_name", type=str, default=None, help="wandb run name")
    parser.add_argument(
        "--vis_epoch",
        type=int,
        default=20,
        help="Create visualization at these epochs. 0 for no logging.",
    )
    parser.add_argument("--save_epoch", type=int, default=0, help="Save model epoch")
    parser.add_argument(
        "--method",
        type=str,
        default="rmix",
        # choices=["rlmix", "rmix", "vanilla", "input", "cutmix", "inputcut", "cutpaste", "cuttopleast"],
        help="Method to train",
    )
    parser.add_argument(
        "--use_fp16", type=str2bool, default=False, help="mixed precision training"
    )
    parser.add_argument(
        "--print_freq", type=int, default=100, help="Batch printing frequency"
    )
    parser.add_argument(
        "--random_patches",
        type=str,
        default="8 16",
        help="Patch size for the Random method. Num_patches = img_size / random_patches",
    )
    parser.add_argument(
        "--use_random_patches",
        type=str2bool,
        default=True,
        help="Choose a random number of patches each step (r-mix).",
    )
    parser.add_argument(
        "--num_action",
        type=int,
        default=10,
        help="Number of percentile values per image",
    )
    parser.add_argument(
        "--env",
        default="VanillaMixupPatchDiscrete",
        choices=["VanillaMixupPatchDiscrete"],
        help="available environment to choose",
    )

    # CNN arguments
    cnn_model = parser.add_argument_group("CNN model arguments")
    cnn_model.add_argument(
        "--cnn_batch_size",
        type=int,
        default=100,
        help="batch size for the CNN model. It should be divisible by the total number of images "
        "(50000 for CIFAR100, 100000 for Tiny-ImageNet)",
    )
    cnn_model.add_argument(
        "--cnn_epoch", type=int, default=300, help="number of epochs to train the model"
    )
    cnn_model.add_argument(
        "--use_scheduler",
        type=str2bool,
        default=True,
        help="whether to use OneCycleLR scheduler (recommended)",
    )
    cnn_model.add_argument(
        "--scheduler",
        type=str,
        default="OneCycleLR",
        choices=["OneCycleLR", "MultiStepLR"],
        help="scheduler name",
    )

    # Scheduler arguments
    one_cycle_lr = parser.add_argument_group("OneCycleLR")
    one_cycle_lr.add_argument(
        "--max_lr", type=float, default=0.29947526988779305, help="Max LR. Will override LR."
    )
    one_cycle_lr.add_argument(
        "--div_factor",
        type=int,
        default=100,
        help="Initial LR, equals max_lr / div_factor",
    )
    one_cycle_lr.add_argument(
        "--pct_start",
        type=float,
        default=0.3,
        help="Percentage of the cycle spent increasing LR,"
        " equals cnn_epoch * pct_start. After that "
        "LR will be gradually decreased",
    )
    one_cycle_lr.add_argument(
        "--final_div_factor",
        type=int,
        default=10000,
        help="Final LR, equals max_lr/final_div_factor",
    )

    step_lr = parser.add_argument_group("MultiStepLR")
    step_lr.add_argument(
        "--milestones",
        type=int,
        default=[150, 225],
        nargs="+",
        help="decay LR at these epochs",
    )
    step_lr.add_argument(
        "--lr_step_gamma",
        type=float,
        default=0.1,
        help="LR decay factor. New LR = LR * step_size_gamma",
    )

    # Agent arguments
    agent = parser.add_argument_group("RL Agent arguments")
    agent.add_argument(
        "--agent_batch_size",
        type=int,
        default=250,
        help="agent batch size. It is suggested " "to be equal to reward_step",
    )
    agent.add_argument(
        "--agent_trajectory_size",
        type=int,
        default=1000,
        help="agent trajectory size. "
        "It is suggested to be a multiple of (num_img / cnn_batch_size)",
    )

    agent.add_argument(
        "--agent_epochs",
        type=int,
        default=15,
        help="number of epochs to update the agent every episode",
    )

    # Environment arguments
    env = parser.add_argument_group("Environment arguments")
    env.add_argument(
        "--reward_step",
        type=int,
        default=1,
        help="number of steps to give the agent the reward.",
    )
    env.add_argument(
        "--reward_scaling", type=float, default=1, help="Reward scaling factor"
    )
    env.add_argument(
        "--shuffle_data",
        type=str2bool,
        default=True,
        help="whether to shuffle the dataset after each epoch",
    )

    # Environment Reward
    reward_env = env.add_argument_group("Reward function design")
    reward_env.add_argument(
        "--grad_sim",
        type=str2bool,
        default=True,
        help="Use cosine similarity between gradients of original and mixed as reward. "
        "Doing so will disable reward_scaling.",
    )

    # Save paths
    save = parser.add_argument_group("Save utilities")
    save.add_argument(
        "--cnn_save_path",
        type=str,
        default="./best_checkpoint" + str(id) + ".pth.tar",
        help="CNN save path",
    )
    save.add_argument(
        "--scheduler_save_path",
        type=str,
        default="./saved_scheduler" + str(id) + ".pt",
        help="scheduler save path",
    )
    save.add_argument(
        "--rl_save_path",
        type=str,
        default="./saved_rl_model" + str(id),
        help="RL agent save path",
    )

    parsed = parser.parse_args()
    parsed.random_patches = list(map(int, parsed.random_patches.split(" ")))
    return parser.parse_args()
