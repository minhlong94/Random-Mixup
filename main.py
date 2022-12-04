import timm
import torchmetrics
import dataloaders.dataloaders
import parse_args
import os.path
import torch
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.ppo import PPO
import torchvision
from model import resnext, preactresnet, wideresnet, resnet
from wandb.integration.sb3 import WandbCallback
import wandb
from policy_network import CNNExtractor
from utils import set_global_seed, exp_schedule
# from torchattacks import PGD, FGSM
from accelerate import Accelerator
import utils
import warnings
import time
from mixup import *
from torch.autograd import Variable


# from gooey import Gooey
#
#
# @Gooey
def main():
    args = parse_args.parse()
    set_global_seed(args.seed)
    args.random_patches = list(map(int, args.random_patches.split(" ")))
    accelerator = Accelerator(fp16=args.use_fp16)

    # Setup wandb
    if args.use_wandb:
        if not args.wandb_offline:
            os.environ["WANDB_MODE"] = "online"
            wandb.login(key=args.wandb_key)
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                sync_tensorboard=True,
            )
            if args.wandb_name is not None:
                wandb.run.name = args.wandb_name
            wandb.config.update(args)
        else:
            os.environ["WANDB_MODE"] = "offline"
            wandb.init()
            wandb.config.update(args)

    num_devices = torch.cuda.device_count()
    print(f"Detect {num_devices} GPU devices.")
    if num_devices > 1:
        warnings.warn("Multi GPU is experimental.")

    environment = __import__("environment")

    train_loader, test_loader, config = dataloaders.dataloaders.load_data(args)
    policy_kwargs = dict(
        features_extractor_class=CNNExtractor,
        features_extractor_kwargs=dict(
            features_dim=256, num_class=config["num_class"]
        ),
        normalize_images=False,
    )

    # Model setup
    if args.model_name == "preactresnet18":
        model = preactresnet.preactresnet18(
            num_classes=config["num_class"], stride=config["stride"]).cuda()
    elif args.model_name == "resnext29_4_24":
        model = resnext.resnext29_4_24(
            num_classes=config["num_class"], stride=config["stride"]).cuda()
    elif args.model_name == "wrn16_8":
        model = wideresnet.wrn16_8(num_classes=config["num_class"], stride=config["stride"]).cuda()
    elif args.model_name == "wrn28_10":
        model = wideresnet.wrn28_10(num_classes=config["num_class"], stride=config["stride"]).cuda()
    elif args.model_name == "resnet50":
        model = resnet.ResNet(dataset="imagenet", depth=50, num_classes=config["num_class"], bottleneck=True).cuda()

    else:
        raise NotImplementedError(
            "Model should be in the list of available models"
        )

    optimizer = torch.optim.SGD(
        model.parameters(), args.lr, momentum=0.9, nesterov=True, weight_decay=1e-4
    )

    if args.use_scheduler:
        if args.scheduler == "OneCycleLR":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                max_lr=args.max_lr,
                div_factor=args.div_factor,
                pct_start=args.pct_start,
                final_div_factor=args.final_div_factor,
                epochs=args.cnn_epoch,
                steps_per_epoch=config["steps_per_epoch"],
            )
        elif args.scheduler == "MultiStepLR":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer,
                milestones=args.milestones,
                gamma=args.lr_step_gamma,
            )
    else:
        scheduler = None

    optimizer, train_loader = accelerator.prepare(optimizer, train_loader)

    # Experimental: multi GPU
    if num_devices > 1:
        model = torch.nn.DataParallel(
            model
        ).cuda()  # We don't know why multi GPU is slower than single GPU
    else:
        model = accelerator.prepare(model)

    ENV_config = dict(
        args=args,
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        accelerator=accelerator,
    )

    if not os.path.isdir("./agent_cp" + str(args.seed)):
        os.mkdir("./agent_cp" + str(args.seed))
    if not os.path.isdir("./vis"):
        os.mkdir("./vis")

    if args.save_epoch != 0:
        callback = WandbCallback(
            model_save_path="./agent_cp" + str(args.seed) + "/", verbose=1
        )
    else:
        callback = None

    PPO_config = dict(
        policy_kwargs=policy_kwargs,
        n_steps=args.agent_trajectory_size,
        batch_size=args.agent_batch_size,
        verbose=1,
        device="cuda",
    )

    print(f"The following configurations are set: {vars(args), config.items()}")
    if args.method == "rlmix":
        env = getattr(environment, args.env)(**ENV_config)
        env = Monitor(env, "log" + str(args.seed))
        print("Running RL method.")
        logger = configure(
            "./agent_log" + str(args.seed) + "/", ["stdout", "tensorboard"]
        )
        agent = PPO(
            "MultiInputPolicy",
            env,
            **PPO_config,
            learning_rate=exp_schedule(args.agent_lr),
            tensorboard_log="tsb_log/tsb" + str(args.seed),
        )
        agent.set_logger(logger=logger)
        agent.learn(
            total_timesteps=args.cnn_epoch * config["steps_per_epoch"],
            callback=callback,
        )

        # torch.save(env.model.state_dict(), args.cnn_save_path)
        # if args.use_scheduler:
        #     torch.save(env.scheduler.state_dict(), args.scheduler_save_path)
        agent.save(args.rl_save_path)

    else:
        env = getattr(environment, args.env)(**ENV_config)
        env = Monitor(env, "log" + str(args.seed))
        print("Running Random method.")
        for i in range(args.cnn_epoch):
            env.reset()
            done = False
            while not done:
                _, _, done, _ = env.step(env.action_space.sample())

        # torch.save(env.model.state_dict(), args.cnn_save_path)


    # if args.dataset == "cifar100":
    #     model = env.model if args.method == "rl-mix" else model
    #     atk = PGD(model, eps=8 / 255, alpha=2 / 255, steps=7)
    #     atk.set_return_type("float")
    #     print("PGD L-Inf with eps 8/255, alpha 2/255, steps 7 attack")
    #     atk.save(
    #         data_loader=test_loader,
    #         save_path="./cifar100_pgd" + str(args.seed) + ".pt",
    #         verbose=True,
    #     )
    #
    #     atk = FGSM(model, eps=8 / 255)
    #     atk.set_return_type("float")  # Save as integer.
    #     print("FGSM with eps 8/255 attack")
    #     atk.save(
    #         data_loader=test_loader,
    #         save_path="./cifar100_fgsm" + str(args.seed) + ".pt",
    #         verbose=True,
    #     )

    if args.use_wandb:
        wandb.save(args.cnn_save_path)


if __name__ == "__main__":
    main()
    exit()
