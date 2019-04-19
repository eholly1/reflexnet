from abc import ABC, abstractmethod
import torch

class Dataset(ABC):

    @property
    @abstractmethod
    def N(self):
        raise NotImplementedError

    @abstractmethod
    def sample(self, batch_size=None, eval=False):
        raise NotImplementedError

class Trainer(ABC):
    """This class orchestrates training, including managing:
        - Train / test frequenceis.
        - Calling gradients / optimizer.
        - Checkpointing.
        - Gathering and saving summaries.
    """
    # TODO(eholly1): Checkpointing.
    # TODO(eholly1): Summaries.

    def __init__(self, model, dataset, optim_cls=torch.optim.Adam, learning_rate=1e-5):
        assert issubclass(type(model), torch.nn.Module)
        self._model = model
        assert issubclass(dataset.__class__, Dataset)
        self._dataset = dataset
        self._optimizer = optim_cls(
            params=self._model.parameters(),
            lr=learning_rate)

    @abstractmethod
    def _inference_and_loss(self, sample_data):
        """Perform inference and compute loss on samples.
        Args:
            sample_data: A sample of data with which to compute loss.
        Returns:
            A scalar loss tensor.
        """
        raise NotImplementedError

    @property
    def model(self):
        return self._model

    def train_and_test(
        self,
        log_dir,
        train_steps,
        eval_every=None,
        log_every=None,
        after_eval_callback=None,
        ):
        if eval_every is None:
            eval_every = train_steps / 20
        if log_every is None:
            log_every = train_steps / 100

        running_loss = 0.0
        steps_since_log = 0.0
        for i in range(train_steps):
            if i % eval_every == 0:
                eval_loss = self._test()
                self.print(i, "Eval loss: ", eval_loss)
                if after_eval_callback is not None:
                    after_eval_callback(i)
            running_loss += self._train()
            steps_since_log += 1.0
            if (i+1) % log_every == 0:
                self.print(i, "Train Loss: ", running_loss / steps_since_log)
                running_loss = 0.0
                steps_since_log = 0.0

    def print(self, i, *args):
        args = ["[{}]\t".format(i)] + list(args)
        print_str = ("{}" * len(args)).format(*args)
        print(print_str)

    def _train(self):
        self._optimizer.zero_grad()
        self._model.train()  # Put model in train mode.
        sample_data = self._dataset.sample()
        loss = self._inference_and_loss(sample_data)
        loss.backward()
        self._optimizer.step()
        return loss

    def _test(self):
        with torch.no_grad():
            self._model.eval()  # Put model in eval mode.
            sample_data = self._dataset.sample(batch_size=float('inf'), eval=True)
            loss = self._inference_and_loss(sample_data)
            return loss