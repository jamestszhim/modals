import random

import numpy as np
import ray
import ray.tune as tune
from modals.setup import create_hparams, create_parser
from modals.trainer import TextModelTrainer
from ray.tune.schedulers import PopulationBasedTraining


class RayModel(tune.Trainable):
    def _setup(self, *args):
        self.trainer = TextModelTrainer(self.config)

    def _train(self):
        print(f'Starting Ray Iteration: {self._iteration}')
        train_acc, valid_acc = self.trainer.run_model(self._iteration)
        test_acc, test_loss = self.trainer._test(self._iteration, mode='test')
        return {'train_acc': train_acc, 'valid_acc': valid_acc, 'test_acc': test_acc}

    def _save(self, checkpoint_dir):
        print(checkpoint_dir)
        path = self.trainer.save_model(checkpoint_dir, self._iteration)
        print(path)
        return path

    def _restore(self, checkpoint_path):
        self.trainer.load_model(checkpoint_path)

    def reset_config(self, new_config):
        self.config = new_config
        self.trainer.reset_config(self.config)
        return True


def search():
    FLAGS = create_parser('search')
    hparams = create_hparams('search', FLAGS)

    # if FLAGS.restore:
    #     train_spec["restore"] = FLAGS.restore

    def explore(config):
        """Custom explore function.

    Args:
      config: dictionary containing ray config params.

    Returns:
      Copy of config with modified augmentation policy.
    """
        new_params = []
        for i, param in enumerate(config["hp_policy"]):
            if random.random() < 0.2:
                new_params.append(random.randint(0, 10))
            else:
                amt = np.random.choice(
                    [0, 1, 2, 3], p=[0.25, 0.25, 0.25, 0.25])
                amt = int(amt)
                if random.random() < 0.5:
                    new_params.append(max(0, param - amt))
                else:
                    new_params.append(min(10, param + amt))
        config["hp_policy"] = new_params
        return config

    ray.init()

    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="valid_acc",
        mode='max',
        perturbation_interval=FLAGS.perturbation_interval,
        custom_explore_fn=explore,
        log_config=True)

    tune.run(
        RayModel,
        name=hparams['ray_name'],
        scheduler=pbt,
        reuse_actors=True,
        verbose=True,
        checkpoint_score_attr="valid_acc",
        checkpoint_freq=FLAGS.checkpoint_freq,
        resources_per_trial={"gpu": FLAGS.gpu, "cpu": FLAGS.cpu},
        stop={"training_iteration": hparams['num_epochs']},
        config=hparams,
        local_dir=FLAGS.ray_dir,
        num_samples=FLAGS.num_samples
    )


if __name__ == "__main__":
    search()
