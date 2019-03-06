import argparse

import torch

from env import R2RPanoBatch, load_features
from eval import Evaluation
from utils import setup, read_vocab, Tokenizer, set_tb_logger, is_experiment, padding_idx, resume_training, save_checkpoint
from trainer import PanoSeq2SeqTrainer
from agents import PanoSeq2SeqAgent
from models import EncoderRNN, SelfMonitoring, SpeakerFollowerBaseline, Regretful


parser = argparse.ArgumentParser(description='PyTorch for Matterport3D Agent with panoramic view and action')
# General options
parser.add_argument('--exp_name', default='experiments_', type=str,
                    help='name of the experiment. \
                        It decides where to store samples and models')
parser.add_argument('--exp_name_secondary', default='', type=str,
                    help='name of the experiment. \
                        It decides where to store samples and models')

# Dataset options
parser.add_argument('--train_vocab', default='tasks/R2R-pano/data/train_vocab.txt',
                    type=str, help='path to training vocab')
parser.add_argument('--trainval_vocab', default='tasks/R2R-pano/data/trainval_vocab.txt',
                    type=str, help='path to training and validation vocab')
parser.add_argument('--img_feat_dir', default='img_features/ResNet-152-imagenet.tsv',
                    type=str, help='path to pre-cached image features')

# Training options
parser.add_argument('--start_epoch', default=1, type=int)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--train_iters_epoch', default=200, type=int,
                    help='number of iterations per epoch')
parser.add_argument('--max_num_epochs', default=300, type=int,
                    help='number of total epochs to run')
parser.add_argument('--eval_every_epochs', default=1, type=int,
                    help='how often do we eval the trained model')
parser.add_argument('--patience', default=3, type=int,
                    help='Number of epochs with no improvement after which learning rate will be reduced.')
parser.add_argument('--min_lr', default=1e-6, type=float,
                    help='A lower bound on the learning rate of all param groups or each group respectively')
parser.add_argument('--seed', default=5, type=int, help='random seed')
parser.add_argument('--train_data_augmentation', default=0, type=int,
                    help='Training with the synthetic data generated with speaker')
parser.add_argument('--epochs_data_augmentation', default=5, type=int,
                    help='Number of epochs for training with data augmentation first')

# General model options
parser.add_argument('--arch', default='regretful', type=str,
                    help='options: regretful | self-monitoring | speaker-baseline')
parser.add_argument('--max_navigable', default=16, type=int,
                    help='maximum number of navigable locations in the dataset is 15 \
                         we add one because the agent can decide to stay at its current location')
parser.add_argument('--use_ignore_index', default=1, type=int, help='ignore target after agent has ended')

# Agent options
parser.add_argument('--follow_gt_traj', default=0, type=int,
                    help='the shortest path to the goal may not match with the instruction if we use student forcing, '
                         'we provide option that the next ground truth viewpoint will try to steer back to the original'
                         'ground truth trajectory')
parser.add_argument('--teleporting', default=1, type=int,
                    help='teleporting: jump directly to next viewpoint, if not rotate and forward until you reach the '
                         'viewpoint with roughly the same heading')
parser.add_argument('--max_episode_len', default=10, type=int, help='maximum length of episode')
parser.add_argument('--feedback_training', default='sample', type=str,
                    help='options: sample | mistake (this is the feedback for training only)')
parser.add_argument('--feedback', default='argmax', type=str,
                    help='options: sample | argmax (this is the feedback for testing only)')
parser.add_argument('--value_loss_weight', default=0.5, type=float,
                    help='the weight applied on the auxiliary value loss')
parser.add_argument('--norm_value', default=0, type=int,
                    help='when using value prediction, do we normalize the distance improvement as value target?')
parser.add_argument('--mse_sum', default=1, type=int,
                    help='when using value prediction, use MSE loss with sum or average across non-navigable directions?')
parser.add_argument('--entropy_weight', default=0.01, type=float,
                    help='weighting for entropy loss')
parser.add_argument('--fix_action_ended', default=1, type=int,
                    help='Action set to 0 if ended. This prevent the model keep getting loss from logit after ended')
parser.add_argument('--monitor_sigmoid', default=0, type=int,
                    help='Use Sigmoid function for progress monitor instead of Tanh')

# Agent rollback options
parser.add_argument('--prevent_oscillation', default=1, type=int,
                    help='Block out the viewpoint that leads to oscillation')
parser.add_argument('--prevent_rollback', default=0, type=int,
                    help='Block out the viewpoint that leads to rollback')
parser.add_argument('--rollback_reset_heading', default=0, type=int,
                    help='We reset the heading if the agent rollback to the previously visited viewpoint')

# Image context
parser.add_argument('--img_feat_input_dim', default=2176, type=int,
                    help='ResNet-152: 2048, if use angle, the input is 2176')
parser.add_argument('--img_fc_dim', default=(128,), nargs="+", type=int)
parser.add_argument('--img_fc_use_batchnorm', default=1, type=int)
parser.add_argument('--img_dropout', default=0.5, type=float)
parser.add_argument('--mlp_relu', default=1, type=int, help='Use ReLu in MLP module')
parser.add_argument('--img_fc_use_angle', default=1, type=int,
                    help='add relative heading and elevation angle into image feature')
parser.add_argument('--progress_marker', default=1, type=int,
                    help='we mark if a viewpoint is visited with progress monitor output and tile it n times')
parser.add_argument('--tiled_len', default=32, type=int, help='tile the marker n times')

# Language model
parser.add_argument('--remove_punctuation', default=0, type=int,
                    help='the original ''encode_sentence'' does not remove punctuation'
                         'we provide an option here.')
parser.add_argument('--reversed', default=1, type=int,
                    help='option for reversing the sentence during encoding')
parser.add_argument('--lang_embed', default='lstm', type=str, help='options: lstm ')
parser.add_argument('--word_embedding_size', default=256, type=int,
                    help='default embedding_size for language encoder.')
parser.add_argument('--rnn_hidden_size', default=256, type=int)
parser.add_argument('--bidirectional', default=0, type=int)
parser.add_argument('--rnn_num_layers', default=1, type=int)
parser.add_argument('--rnn_dropout', default=0.5, type=float)
parser.add_argument('--max_cap_length', default=80, type=int, help='maximum length of captions')

# Evaluation options
parser.add_argument('--eval_only', default=0, type=int,
                    help='No training. Resume from a model and run evaluation')
parser.add_argument('--test_submission', default=0, type=int,
                    help='No training. Resume from a model and run testing for submission')

# Output options
parser.add_argument('--results_dir',
                    default='tasks/R2R-pano/results/',
                    type=str, help='where to save the output results for computing accuracy')
parser.add_argument('--resume', default='', type=str,
                    help='two options for resuming the model: latest | best')
parser.add_argument('--checkpoint_dir',
                    default='tasks/R2R-pano/checkpoints/pano-seq2seq/',
                    type=str, help='where to save trained models')
parser.add_argument('--tensorboard', default=1, type=int,
                    help='Use TensorBoard for loss visualization')
parser.add_argument('--log_dir',
                    default='tensorboard_logs/pano-seq2seq',
                    type=str, help='path to tensorboard log files')

def main(opts):
    # set manual_seed and build vocab
    setup(opts, opts.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create a batch training environment that will also preprocess text
    vocab = read_vocab(opts.train_vocab)
    tok = Tokenizer(opts.remove_punctuation == 1, opts.reversed == 1, vocab=vocab, encoding_length=opts.max_cap_length)

    # create language instruction encoder
    encoder_kwargs = {
        'opts': opts,
        'vocab_size': len(vocab),
        'embedding_size': opts.word_embedding_size,
        'hidden_size': opts.rnn_hidden_size,
        'padding_idx': padding_idx,
        'dropout_ratio': opts.rnn_dropout,
        'bidirectional': opts.bidirectional == 1,
        'num_layers': opts.rnn_num_layers
    }
    print('Using {} as encoder ...'.format(opts.lang_embed))
    if 'lstm' in opts.lang_embed:
        encoder = EncoderRNN(**encoder_kwargs)
    else:
        raise ValueError('Unknown {} language embedding'.format(opts.lang_embed))
    print(encoder)

    # create policy model
    policy_model_kwargs = {
        'opts':opts,
        'img_fc_dim': opts.img_fc_dim,
        'img_fc_use_batchnorm': opts.img_fc_use_batchnorm == 1,
        'img_dropout': opts.img_dropout,
        'img_feat_input_dim': opts.img_feat_input_dim,
        'rnn_hidden_size': opts.rnn_hidden_size,
        'rnn_dropout': opts.rnn_dropout,
        'max_len': opts.max_cap_length,
        'max_navigable': opts.max_navigable
    }

    if opts.arch == 'regretful':
        model = Regretful(**policy_model_kwargs)
    elif opts.arch == 'self-monitoring':
        model = SelfMonitoring(**policy_model_kwargs)
    elif opts.arch == 'speaker-baseline':
        model = SpeakerFollowerBaseline(**policy_model_kwargs)
    else:
        raise ValueError('Unknown {} model for seq2seq agent'.format(opts.arch))
    print(model)

    encoder = encoder.to(device)
    model = model.to(device)

    params = list(encoder.parameters()) + list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=opts.learning_rate)

    # optionally resume from a checkpoint
    if opts.resume:
        model, encoder, optimizer, best_success_rate = resume_training(opts, model, encoder, optimizer)

    # if a secondary exp name is specified, this is useful when resuming from a previous saved
    # experiment and save to another experiment, e.g., pre-trained on synthetic data and fine-tune on real data
    if opts.exp_name_secondary:
        opts.exp_name += opts.exp_name_secondary

    feature, img_spec = load_features(opts.img_feat_dir)

    if opts.test_submission:
        assert opts.resume, 'The model was not resumed before running for submission.'
        test_env = ('test', (R2RPanoBatch(opts, feature, img_spec, batch_size=opts.batch_size,
                                 splits=['test'], tokenizer=tok), Evaluation(['test'])))
        agent_kwargs = {
            'opts': opts,
            'env': test_env[1][0],
            'results_path': "",
            'encoder': encoder,
            'model': model,
            'feedback': opts.feedback
        }
        agent = PanoSeq2SeqAgent(**agent_kwargs)
        # setup trainer
        trainer = PanoSeq2SeqTrainer(opts, agent, optimizer)
        epoch = opts.start_epoch - 1
        trainer.eval(epoch, test_env)
        return

    # set up R2R environments
    if not opts.train_data_augmentation:
        train_env = R2RPanoBatch(opts, feature, img_spec, batch_size=opts.batch_size, seed=opts.seed,
                                 splits=['train'], tokenizer=tok)
    else:
        train_env = R2RPanoBatch(opts, feature, img_spec, batch_size=opts.batch_size, seed=opts.seed,
                                 splits=['synthetic'], tokenizer=tok)

    val_envs = {split: (R2RPanoBatch(opts, feature, img_spec, batch_size=opts.batch_size,
                                     splits=[split], tokenizer=tok), Evaluation([split]))
                for split in ['val_seen', 'val_unseen']}

    # create agent
    agent_kwargs = {
        'opts': opts,
        'env': train_env,
        'results_path': "",
        'encoder': encoder,
        'model': model,
        'feedback': opts.feedback
    }
    agent = PanoSeq2SeqAgent(**agent_kwargs)

    # setup trainer
    trainer = PanoSeq2SeqTrainer(opts, agent, optimizer, opts.train_iters_epoch)

    if opts.eval_only:
        success_rate = []
        for val_env in val_envs.items():
            success_rate.append(trainer.eval(opts.start_epoch - 1, val_env, tb_logger=None))
        return

    # set up tensorboard logger
    tb_logger = set_tb_logger(opts.log_dir, opts.exp_name, opts.resume)

    best_success_rate = best_success_rate if opts.resume else 0.0
    for epoch in range(opts.start_epoch, opts.max_num_epochs + 1):
        trainer.train(epoch, train_env, tb_logger)

        if epoch % opts.eval_every_epochs == 0:
            success_rate = []
            for val_env in val_envs.items():
                success_rate.append(trainer.eval(epoch, val_env, tb_logger))

            success_rate_compare = success_rate[1]

            if is_experiment():
                # remember best val_seen success rate and save checkpoint
                is_best = success_rate_compare >= best_success_rate
                best_success_rate = max(success_rate_compare, best_success_rate)
                print("--> Highest val_unseen success rate: {}".format(best_success_rate))

                # save the model if it is the best so far
                save_checkpoint({
                    'opts': opts,
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'encoder_state_dict': encoder.state_dict(),
                    'best_success_rate': best_success_rate,
                    'optimizer': optimizer.state_dict(),
                    'max_episode_len': opts.max_episode_len,
                }, is_best, checkpoint_dir=opts.checkpoint_dir, name=opts.exp_name)

        if opts.train_data_augmentation and epoch == opts.epochs_data_augmentation:
            train_env = R2RPanoBatch(opts, feature, img_spec, batch_size=opts.batch_size, seed=opts.seed,
                                     splits=['train'], tokenizer=tok)

    print("--> Finished training")


if __name__ == '__main__':
    opts = parser.parse_args()
    main(opts)