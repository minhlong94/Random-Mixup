import torch
import torchmetrics
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from gym import spaces
from typing import Union
from torch.autograd import Variable
from mixup import mix_topleast_choose_top, mixup_data, mixup_criterion, cut_top_and_paste, cut_topleast_and_paste, \
    mix_topleast_choose_least
from utils import accuracy, AverageMeter, convert_secs2time
from collections import Counter
import wandb
from accelerate import Accelerator
import time


class VanillaMixupPatchDiscrete(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        args,
        config,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Union[torch.optim.Optimizer, None],
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        accelerator: Accelerator,
    ):
        """
        Training environment, compatible with Gym API.
        The __init__ will setup the environment.


        :param
            args: ArgParse object
            config: dict
            model: torch.nn.Module, the classifier
            optimizer: torch.optim.Optimizer, the optimizer
            scheduler: Union[torch.optim.Optimizer, None]: the scheduler, optional
            train_dataset: torch.utils.data.Dataset, the training dataset
            test_dataset: torch.utils.data.Dataset, the test dataset

        """

        self.args = args
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_dataloader
        self.test_loader = test_dataloader
        self.accelerator = accelerator
        self.best_test_top1 = 0.0

        self.action_space = spaces.MultiDiscrete(
            [args.num_action] * args.cnn_batch_size
        )
        self.action_space_mapper = np.linspace(0.0, 0.99, num=args.num_action)
        self.action_counter = Counter()
        self.action_list: list = []
        self.step_counter: int = 0
        obs_space = dict(
            saliency=spaces.Box(
                0.0,
                1.0,
                shape=(args.cnn_batch_size, args.num_patches, args.num_patches),
            ),
            logits=spaces.Box(
                -np.inf, np.inf, shape=(args.cnn_batch_size, config["num_class"])
            ),
        )
        self.observation_space = spaces.Dict(obs_space)

        # Setup loss function, optimizer and scheduler
        self.CELoss_saliency = nn.CrossEntropyLoss(reduction="none").cuda()  # For saliency
        self.BCELoss_model = nn.BCELoss().cuda()  # For one-hot targets
        self.patch_size = self.config["size"] // args.num_patches

        # Utils
        self.episode_counter = 0
        self.reward_avgmeter = AverageMeter()
        self.epoch_time = AverageMeter()
        self.batch_time = AverageMeter()

    def grad_sim(self, tensorA: torch.Tensor, tensorB: torch.Tensor):
        reward = (
            F.cosine_similarity(
                torch.flatten(tensorA, start_dim=1),
                torch.flatten(tensorB, start_dim=1),
                dim=1,
                eps=1e-12,
            )
            .squeeze(0)
            .cpu()
            .numpy()
            .item()
        )
        # print(f"Reward: {reward} at step: {self.step_counter}, of episode: {self.episode_counter}")
        return reward

    def step(self, action: np.ndarray):
        """
        Take a step.
        1) Actions are repeated if it does not match patch size
        2) The image is mixed based on selected actions
        3) Train the model with the mixed images
        4) Get the reward as the gradient similarity between the mixed images and the original images
        5) Try loading the next batch:
            If successful, return the saliency of that batch
            If failed, the train dataset reaches the end, then return a np.zeros array. The env will be resetted.
                The np.zeros array will NOT be used to train the agent.
        6) Return a tuple of four: state: dict(saliency: np.ndarray, logits: np.ndarray),
            reward: float, done: bool, info: dict

        Reference:
            https://github.com/snu-mllab/Co-Mixup/blob/5464a8b98bc4f5fa2b8152ce2ba3e608b69afe9d/main.py#L374
            https://discuss.pytorch.org/t/cross-entropy-with-one-hot-targets/13580
            https://discuss.pytorch.org/t/applying-cross-entropy-loss-on-one-hot-targets/87464

        :param action: np.ndarray of actions
        :return
            state: dict(saliency=np.ndarray, logits=np.ndarray), saliency and logits of the next batch
            reward: float
            done: bool
            info: dict
        """

        # Get actions and map it to its corresponding values
        batch_start_time = time.time()
        actions = []
        for idx, act in enumerate(action):
            self.action_list.append(act)
            self.action_counter[act] += 1
            actions.append(self.action_space_mapper[act])
        top_k_patch = torch.as_tensor(actions, device="cuda").float()
        # print(f"Top-k patches: {top_k_patch}", end="\t")
        self.step_counter += 1

        # Train the model
        self.model.train()
        inputs, targets = self.train_batch
        # Input Mixup, patch level
        method = np.random.choice(self.args.method.split(" "))
        if method == "rlmix" or method == "rmix":
            with torch.no_grad():
                if self.args.use_random_patches:
                    mixed_inputs, mixed_targets = mix_topleast_choose_top(
                        inputs,
                        targets,
                        self.mixup_saliency,
                        top_k_patch,
                        self.args,
                        self.config,
                        self.config["size"] // self.mixup_saliency.shape[-1],
                        self.episode_counter,
                        self.vis_flag,
                    )
                else:
                    mixed_inputs, mixed_targets = mix_topleast_choose_top(
                        inputs,
                        targets,
                        self.saliency,
                        top_k_patch,
                        self.args,
                        self.config,
                        self.patch_size,
                        self.episode_counter,
                        self.vis_flag,
                    )
            # Train the model
            mixed_inputs, mixed_targets = Variable(
                mixed_inputs, requires_grad=True
            ), Variable(mixed_targets)
            outputs = self.model(mixed_inputs.float())

            loss = torch.mean(
                torch.sum(-mixed_targets * nn.LogSoftmax(-1)(outputs), dim=1)
            )

        elif method == "input":
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets,
                                                           self.args.dirichlet_alpha, True)
            targets_a, targets_b = map(Variable, (targets_a, targets_b))
            mixed_inputs = Variable(inputs, requires_grad=True)
            outputs = self.model(mixed_inputs)
            criterion = torch.nn.CrossEntropyLoss().cuda()
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            # loss = torch.mean(
            #     torch.sum(-targets * nn.LogSoftmax(-1)(outputs), dim=1)
            # )

        elif method == "cutmix":
            criterion = torch.nn.CrossEntropyLoss().cuda()
            prob = np.random.rand(1)
            if prob < 0.5:
                # generate mixed sample
                lam = np.random.beta(self.args.dirichlet_alpha, self.args.dirichlet_alpha)
                rand_index = torch.randperm(inputs.size()[0]).cuda()
                target_a = targets
                target_b = targets[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
                inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
                # compute output
                mixed_inputs = Variable(inputs, requires_grad=True)
                outputs = self.model(mixed_inputs)
                loss = criterion(outputs, target_a) * lam + criterion(outputs, target_b) * (1. - lam)
            else:
                mixed_inputs = Variable(inputs, requires_grad=True)
                outputs = self.model(mixed_inputs)
                loss = criterion(outputs, targets)


        elif method == "vanilla":
            mixed_inputs = Variable(inputs, requires_grad=True)
            outputs = self.model(mixed_inputs)
            loss = torch.mean(
                torch.sum(-targets * nn.LogSoftmax(-1)(outputs), dim=1)
            )
        else:
            raise Exception("Invalid method")

        self.vis_flag = False
        self.train_losses.update(loss.item(), inputs.size(0))
        self.train_top1.update(outputs, targets)
        self.train_top5.update(outputs, targets)

        # Compute gradients and do backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.args.use_scheduler:
            if self.args.scheduler == "OneCycleLR":
                self.scheduler.step()

        info = {}
        mix_saliency = torch.sqrt(torch.mean(mixed_inputs.grad ** 2, dim=1))

        # Reward step
        if self.step_counter % self.args.reward_step == 0:
            if self.args.grad_sim:
                with torch.no_grad():
                    mix_saliency = F.avg_pool2d(
                        mix_saliency, kernel_size=self.patch_size
                    )
                    reward = self.grad_sim(
                        self.original_saliency.unsqueeze(0), mix_saliency.unsqueeze(0)
                    )
        else:
            reward = 0

        reward /= self.args.reward_scaling

        self.reward_avgmeter.update(reward, 1)

        if self.step_counter % self.args.print_freq == 0:
            print(
                f"\n\nEpisode: {self.episode_counter} | Step: {self.step_counter} | Train loss: "
                f"{self.train_losses.avg:.5f} | Train top1 acc: {self.train_top1.compute().item() * 100:.3f}% | "
                f"Train top5 acc: {self.train_top5.compute().item() * 100:.3f}% | Batch time: {self.batch_time.avg:.5f}",
                end=" ",
            )

        # Load the next batch
        try:
            self.train_batch = next(self.train_loader_iter)
        # if reaches the end
        except StopIteration:
            # print(torch.mean(torch.mean(torch.stack(self.label_tensor), dim=1), dim=0))
            done = True
            state = np.zeros(
                (self.args.cnn_batch_size, self.args.num_patches, self.args.num_patches)
            )  # Just return a random array
            logits = np.zeros((self.config["num_class"],))
            test_losses, test_top1, test_top5 = self.test_model()
            self.train_top1 = self.train_top1.compute().item() * 100
            self.train_top5 = self.train_top5.compute().item() * 100
            test_top1 = test_top1.compute().item() * 100
            test_top5 = test_top5.compute().item() * 100

            if self.args.use_wandb:
                wandb.log(
                    {
                        "train_loss": self.train_losses.avg,
                        "train_top1": self.train_top1,
                        "test_loss": test_losses.avg,
                        "test_top1": test_top1,
                        "test_top5": test_top5,
                        "reward_avg": self.reward_avgmeter.avg,
                        "reward_sum": self.reward_avgmeter.sum,
                        "actions": wandb.Histogram(self.action_list),
                    }
                )

            if self.args.use_scheduler:
                if self.args.scheduler == "MultiStepLR":
                    self.scheduler.step()

            self.epoch_time.update(time.time() - self.start_time)
            need_hour, need_mins, need_secs = convert_secs2time(
                self.epoch_time.avg * (self.args.cnn_epoch - self.episode_counter)
            )
            need_time = "| Estimated Time Left: {:02d}:{:02d}:{:02d}\n\n".format(
                need_hour, need_mins, need_secs
            )
            print(
                "\n\nEpisode: {} | Step: {} | Train loss: {:.5f} | Train top1 acc: {:.3f}% | "
                "Train top5 acc: {:.3f}% | Test Loss: {:.5f} | Test top1 acc: {:.3f}% | Test top5 acc: {:.3f}% |"
                " Epoch time: {:.5f}".format(
                    self.episode_counter,
                    self.step_counter,
                    self.train_losses.avg,
                    self.train_top1,
                    self.train_top5,
                    test_losses.avg,
                    test_top1,
                    test_top5,
                    self.epoch_time.avg,
                ),
                end=" ",
            )
            print(need_time)
            if test_top1 > self.best_test_top1:
                file = {
                    "model": self.model.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "test_top1": test_top1,
                    "test_top5": test_top5,
                    "args": self.args,
                    "config": self.config,
                    "epoch": self.episode_counter
                }
                torch.save(file, self.args.cnn_save_path)
                self.best_test_top1 = test_top1
                wandb.log({"best_test_top1": self.best_test_top1})
        else:
            done = False
            # Compute saliency map of the next batch
            state, logits = self.compute_saliency()

        # Batch time
        self.batch_time.update(time.time() - batch_start_time)

        return dict(saliency=state, logits=logits), reward, done, info

    def test_model(self):
        """
        Test the model on the full and clean test set

        :return:
            test_losses: AverageMeter
            test_top1: AverageMeter
            test_top5: AverageMeter
        """
        with torch.no_grad():
            self.model.eval()
            test_losses = AverageMeter()
            test_top1 = torchmetrics.Accuracy(top_k=1).cuda()
            test_top5 = torchmetrics.Accuracy(top_k=5).cuda()

            for batch_idx, (inputs_test, targets_test) in enumerate(self.test_loader):
                inputs_test, targets_test = inputs_test.cuda(), targets_test.cuda()
                targets_test_one_hot = F.one_hot(
                    targets_test, self.config["num_class"]
                ).float()
                outputs = self.model(inputs_test.float())
                # loss = self.BCELoss_model(
                #     nn.Softmax(dim=1).cuda()(outputs), targets_test_one_hot
                # )
                loss = torch.mean(
                    torch.sum(-targets_test_one_hot * nn.LogSoftmax(-1)(outputs), dim=-1)
                )
                test_losses.update(loss.item(), inputs_test.size(0))
                test_top1.update(outputs, targets_test)
                test_top5.update(outputs, targets_test)
            return test_losses, test_top1, test_top5

    def compute_saliency(self):
        """
        Compute the saliency as the L2 norm of gradients across input channels.

        :return:
        """
        self.model.train()
        inputs_next, targets_next = self.train_batch

        input_var = Variable(inputs_next, requires_grad=True)
        target_var = Variable(targets_next)
        output_logits = self.model(input_var)

        loss_batch_saliency = (
            2
            * self.CELoss_saliency(output_logits, target_var)
            / self.config["num_class"]
        )
        loss_batch_mean_saliency = torch.mean(loss_batch_saliency, dim=0)
        self.optimizer.zero_grad()
        loss_batch_mean_saliency.backward()
        saliency = torch.sqrt(
            torch.mean(input_var.grad ** 2, dim=1)
        )  # L2 norm of grad across channels

        with torch.no_grad():
            # Downsample
            agent_saliency = F.avg_pool2d(saliency, kernel_size=self.patch_size)
            if self.args.use_random_patches:
                mixup_saliency = F.avg_pool2d(
                    saliency, kernel_size=np.random.choice(self.args.random_patches)
                )

                # Normalize
                mixup_saliency = mixup_saliency / mixup_saliency.reshape(
                    mixup_saliency.shape[0], -1
                ).sum(1).reshape(mixup_saliency.shape[0], 1, 1)

                self.mixup_saliency = mixup_saliency.clone().detach()


            self.original_saliency = (
                agent_saliency.clone().detach()
            )  # Save the original grad for gradient similarity reward

            # The agent requires a fixed size, so this is just for the agent

            agent_saliency = agent_saliency / agent_saliency.reshape(
                agent_saliency.shape[0], -1
            ).sum(1).reshape(agent_saliency.shape[0], 1, 1)

            self.saliency = (
                agent_saliency.clone().detach()
            )  # Stores the current observation
            state = agent_saliency.clone().detach().cpu().numpy()

        return state, output_logits.detach().cpu().numpy()

    def reset(self):
        """
        Reset the env.

        Train dataset will be re-initialized as a Iterable object.
        Get the first batch, compute saliency and logits, then return it as the first observation.

        :return:
            state: dict(saliency: np.ndarray, logits: np.ndarray)
        """
        print(self.action_counter)
        self.episode_counter += 1
        self.start_time = time.time()
        self.label_tensor = []

        # Save visualizations
        if (
            self.args.vis_epoch != 0 and self.episode_counter % self.args.vis_epoch == 0
        ) or (self.args.vis_epoch != 0 and self.episode_counter == 1):
            self.vis_flag = True
        elif self.args.vis_epoch == 0:
            self.vis_flag = False

        # Initialize data loader
        self.train_loader_iter = iter(self.train_loader)
        self.train_batch = next(self.train_loader_iter)

        # Loss value and metrics (accuracy)
        self.train_losses = AverageMeter()
        self.train_top1 = torchmetrics.Accuracy(top_k=1).cuda()
        self.train_top5 = torchmetrics.Accuracy(top_k=5).cuda()

        # Compute the saliency map of the first batch
        state, logits = self.compute_saliency()
        return dict(saliency=state, logits=logits)


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
