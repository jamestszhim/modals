from modals.trainer import TextModelTrainer
from modals.setup import create_parser, create_hparams


def main(FLAGS, hparams):
    start_epoch = 0
    trainer = TextModelTrainer(hparams, FLAGS.name)

    if FLAGS.restore is not None:
        start_epoch, _ = trainer.load_model(FLAGS.restore)

    for e in range(start_epoch+1, hparams['num_epochs']+1):
        trainer.run_model(e)

        if e % 20 == 0:
            # print(hparams)
            trainer.save_checkpoint(hparams['checkpoint_dir'], e)
            trainer._test(e, 'test')

    trainer.save_checkpoint(hparams['checkpoint_dir'], e)
    trainer._test(hparams['num_epochs'], 'test')


if __name__ == "__main__":
    FLAGS = create_parser('train')
    hparams = create_hparams('train', FLAGS)
    main(FLAGS, hparams)
