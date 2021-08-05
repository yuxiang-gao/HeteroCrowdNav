import logging
import copy
from abc import ABC, abstractmethod

import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import optimizer
from torch.utils.data import DataLoader

from tqdm import tqdm


class Trainer(ABC):
    def __init__(
        self, model, memory, device, batch_size, writer, optimizer_name
    ):
        self.model = model
        self.memory = memory
        self.device = device
        self.batch_size = batch_size
        self.optimizer_name = optimizer_name
        self.writer = writer

        self.criterion = nn.MSELoss().to(device)

        self.data_loader = None
        self.optimizer = None
        self.target_model = None

    def update_target_model(self, target_model):
        self.target_model = copy.deepcopy(target_model)

    @abstractmethod
    def set_learning_rate(self, learning_rate):
        pass

    @abstractmethod
    def optimize_epoch(self, num_epochs):
        pass

    @abstractmethod
    def optimize_batch(self, num_batches, episode):
        pass


class VNRLTrainer(Trainer):
    def __init__(
        self, model, memory, device, batch_size, writer, optimizer_name="Adam"
    ):
        """
        Train the trainable model of a policy
        """
        super().__init__(
            model, memory, device, batch_size, writer, optimizer_name
        )

        # for value update
        self.gamma = 0.9
        self.time_step = 0.25
        self.v_pref = 1

    def set_learning_rate(self, learning_rate):
        optimizers = {
            "SGD": optim.Adam(self.model.parameters(), lr=learning_rate),
            "Adam": optim.SGD(
                self.model.parameters(), lr=learning_rate, momentum=0.9
            ),
        }
        self.optimizer = optimizers.get(self.optimizer_name)
        logging.info(
            "Lr: {}  with {} optimizer for parameters [{}]".format(
                learning_rate,
                " ".join(
                    [name for name, param in self.model.named_parameters()]
                ),
                self.optimizer_name,
            )
        )

    def optimize_epoch(self, num_epochs):
        if self.optimizer is None:
            raise ValueError("Learning rate is not set!")
        if self.data_loader is None:
            self.data_loader = DataLoader(
                self.memory, self.batch_size, shuffle=True
            )
        average_epoch_loss = 0
        loop = tqdm(range(num_epochs), desc="Imitate", colour="blue")
        for epoch in loop:
            epoch_loss = 0
            pbar = tqdm(
                self.data_loader,
                desc=f"Epoch [{epoch}/{num_epochs}]",
                colour="green",
                leave=False,
            )
            for inputs, values, _, _ in pbar:
                inputs = Variable(inputs)
                values = Variable(values)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, values)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.data.item()
                # pbar.set_description(f"Epoch [{epoch}/{num_epochs}]")
                pbar.set_postfix(loss=loss.data.item())

            average_epoch_loss = epoch_loss / len(self.memory)
            self.writer.add_scalar(
                "IL/average_epoch_loss", average_epoch_loss, epoch
            )
            loop.set_postfix(loss=average_epoch_loss)
            logging.debug(
                "Average loss in epoch %d: %.2E", epoch, average_epoch_loss
            )

        return average_epoch_loss

    def optimize_batch(self, num_batches, episode=None):
        if self.optimizer is None:
            raise ValueError("Learning rate is not set!")
        if self.data_loader is None:
            self.data_loader = DataLoader(
                self.memory, self.batch_size, shuffle=True
            )
        losses = 0
        pbar = tqdm(
            self.data_loader, total=num_batches, leave=False, colour="green"
        )
        # batch_count = 0
        for batch_count, (inputs, values, rewards, next_states) in enumerate(
            pbar
        ):
            inputs = Variable(inputs)
            values = Variable(values)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            gamma_bar = pow(self.gamma, self.time_step * self.v_pref)
            target_values = rewards + gamma_bar * self.target_model(next_states)

            loss = self.criterion(outputs, target_values)
            loss.backward()
            self.optimizer.step()
            losses += loss.data.item()

            pbar.set_description(f"Episode [{episode}]")
            pbar.set_postfix(loss=loss.data.item())

            if batch_count > num_batches:
                break

        average_loss = losses / num_batches
        self.writer.add_scalar("RL/average_loss", average_loss, episode)
        logging.debug("Average loss : %.2E", average_loss)

        return average_loss
