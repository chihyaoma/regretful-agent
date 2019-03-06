import time
import math
import numpy as np

import torch
from utils import AverageMeter, load_datasets

class PanoSeq2SeqTrainer():
    """Trainer for training and validation process"""
    def __init__(self, opts, agent, optimizer, train_iters_epoch=100):
        self.opts = opts
        self.agent = agent
        self.optimizer = optimizer
        self.train_iters_epoch = train_iters_epoch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, epoch, train_env, tb_logger=None):
        batch_time = AverageMeter()
        losses = AverageMeter()
        dists = AverageMeter()
        movements = AverageMeter()
        val_losses = AverageMeter()
        val_acces = AverageMeter()

        print('Training on {} env ...'.format(train_env.splits[0]))
        # switch to train mode
        self.agent.env = train_env
        self.agent.encoder.train()
        self.agent.model.train()

        self.agent.feedback = self.opts.feedback_training
        self.agent.value_loss = None
        self.agent.val_acc = None
        self.agent.rollback_attn = None

        # load dataset path for computing ground truth distance
        self.agent.gt = {}
        for item in load_datasets(train_env.splits, self.opts):
            self.agent.gt[item['path_id']] = item

        success_count, rollback_success_count, rollback_count, oscillating_success_count, oscillating_count = 0, 0, 0, 0, 0
        end = time.time()
        for iter in range(1, self.train_iters_epoch + 1):
            # roll out the agent
            if self.opts.arch == 'regretful':
                loss, traj = self.agent.rollout_regret()
            elif self.opts.arch == 'self-monitoring':
                loss, traj = self.agent.rollout_monitor()
            elif self.opts.arch == 'speaker-baseline':
                loss, traj = self.agent.rollout()
            else:
                raise NotImplementedError()

            dist_from_goal = np.mean(self.agent.dist_from_goal)
            movement = np.mean(self.agent.traj_length)

            losses.update(loss.item(), self.opts.batch_size)
            dists.update(dist_from_goal, self.opts.batch_size)
            movements.update(movement, self.opts.batch_size)

            if self.agent.value_loss is not None:
                val_losses.update(self.agent.value_loss.item(), self.opts.batch_size)

            if self.agent.val_acc is not None:
                val_acces.update(np.mean(self.agent.val_acc), self.opts.batch_size)

            # zero the gradients before backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if self.agent.rollback_attn is not None:
                if iter == 1:
                    rollback_attn = self.agent.rollback_attn
                else:
                    rollback_attn = np.concatenate((rollback_attn, self.agent.rollback_attn), axis=1)

            if tb_logger and iter % 10 == 0:
                current_iter = iter + (epoch - 1) * self.train_iters_epoch
                tb_logger.add_scalar('train/loss_train', loss, current_iter)
                tb_logger.add_scalar('train/dist_from_goal', dist_from_goal, current_iter)
                tb_logger.add_scalar('train/movements', movement, current_iter)
                if self.agent.value_loss is not None:
                    tb_logger.add_scalar('train/value_loss', self.agent.value_loss, current_iter)

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, iter, self.train_iters_epoch, batch_time=batch_time,
                loss=losses))

            success_count, rollback_success_count, rollback_count, oscillating_success_count, oscillating_count = \
                count_rollback_success(success_count, rollback_success_count, rollback_count, oscillating_success_count,
                                       oscillating_count, traj)

        if tb_logger:
            tb_logger.add_scalar('epoch/learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
            tb_logger.add_scalar('epoch/train/loss', losses.avg, epoch)
            tb_logger.add_scalar('epoch/train/dist_from_goal', dists.avg, epoch)
            tb_logger.add_scalar('epoch/train/movements', movements.avg, epoch)
            if self.agent.value_loss is not None:
                tb_logger.add_scalar('epoch/train/val_loss', val_losses.avg, epoch)
            if self.agent.val_acc is not None:
                tb_logger.add_scalar('epoch/train/val_acc', val_acces.avg, epoch)
            if self.agent.rollback_attn is not None:
                for step in range(self.opts.max_episode_len):
                    tb_logger.add_histogram('epoch_train/rollback_attn_{}'.format(step), rollback_attn[step], epoch)
            tb_logger.add_scalar('rollback_oscillation/train/rollback', rollback_count / len(train_env.data), epoch)
            tb_logger.add_scalar('rollback_oscillation/train/rollback_SR', rollback_success_count / len(train_env.data),
                                 epoch)
            tb_logger.add_scalar('rollback_oscillation/train/oscillating', oscillating_count / len(train_env.data),
                                 epoch)
            tb_logger.add_scalar('rollback_oscillation/train/oscillating_SR', oscillating_success_count / len(train_env.data),
                                 epoch)


    def eval(self, epoch, val_env, tb_logger=None):
        batch_time = AverageMeter()
        losses = AverageMeter()
        dists = AverageMeter()
        movements = AverageMeter()
        val_losses = AverageMeter()
        val_acces = AverageMeter()

        env_name, (env, evaluator) = val_env

        print('Evaluating on {} env ...'.format(env_name))

        self.agent.env = env
        self.agent.env.reset_epoch()
        self.agent.model.eval()
        self.agent.encoder.eval()
        self.agent.feedback = self.opts.feedback
        self.agent.value_loss = None
        self.agent.val_acc = None
        self.agent.rollback_attn = None

        # load dataset path for computing ground truth distance
        self.agent.gt = {}
        for item in load_datasets([env_name]):
            self.agent.gt[item['path_id']] = item
        val_iters_epoch = math.ceil(len(env.data) / self.opts.batch_size)
        self.agent.results = {}
        looped = False
        iter = 1
        success_count, rollback_success_count, rollback_count, oscillating_success_count, oscillating_count = 0, 0, 0, 0, 0

        with torch.no_grad():
            end = time.time()
            while True:

                # roll out the agent
                if self.opts.arch == 'regretful':
                    loss, traj = self.agent.rollout_regret()
                elif self.opts.arch == 'self-monitoring':
                    loss, traj = self.agent.rollout_monitor()
                elif self.opts.arch == 'speaker-baseline':
                    loss, traj = self.agent.rollout()
                else:
                    raise NotImplementedError()

                dist_from_goal = np.mean(self.agent.dist_from_goal)
                movement = np.mean(self.agent.traj_length)

                losses.update(loss.item(), self.opts.batch_size)
                dists.update(dist_from_goal, self.opts.batch_size)
                movements.update(movement, self.opts.batch_size)
                if self.agent.value_loss is not None:
                    val_losses.update(self.agent.value_loss.item(), self.opts.batch_size)
                if self.agent.val_acc is not None:
                    val_acces.update(np.mean(self.agent.val_acc), self.opts.batch_size)

                if tb_logger and iter % 10 == 0:
                    current_iter = iter + (epoch - 1) * val_iters_epoch
                    tb_logger.add_scalar('{}/loss'.format(env_name), loss, current_iter)
                    tb_logger.add_scalar('{}/dist_from_goal'.format(env_name), dist_from_goal, current_iter)
                    tb_logger.add_scalar('{}/movements'.format(env_name), movement, current_iter)
                    if self.agent.value_loss is not None:
                        tb_logger.add_scalar('{}/val_loss'.format(env_name), self.agent.value_loss, current_iter)

                success_count, rollback_success_count, rollback_count, oscillating_success_count, oscillating_count = \
                    count_rollback_success(success_count, rollback_success_count, rollback_count, oscillating_success_count, oscillating_count, traj)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if self.agent.rollback_attn is not None:
                    if iter == 1:
                        rollback_attn = self.agent.rollback_attn
                    else:
                        rollback_attn = np.concatenate((rollback_attn, self.agent.rollback_attn), axis=1)

                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    epoch, iter, val_iters_epoch, batch_time=batch_time,
                    loss=losses))

                # write into results
                for traj_ in traj:
                    if traj_['instr_id'] in self.agent.results:
                        looped = True
                    else:
                        result = {
                            'path': traj_['path'],
                            'distance': traj_['distance'],
                            'img_attn': traj_['img_attn'],
                            'ctx_attn': traj_['ctx_attn'],
                            'rollback_forward_attn': traj_['rollback_forward_attn'],
                            'value': traj_['value'],
                            'viewpoint_idx': traj_['viewpoint_idx'],
                            'navigable_idx': traj_['navigable_idx']
                        }
                        self.agent.results[traj_['instr_id']] = result
                if looped:
                    break
                iter += 1


        print('============================')
        print('success rate: {}'.format(success_count / len(env.data)))
        print('rollback rate: {}'.format(rollback_count / len(env.data)))
        print('rollback success rate: {}'.format(rollback_success_count / len(env.data)))
        print('oscillating rate: {}'.format(oscillating_count / len(env.data)))
        print('oscillating success rate: {}'.format(oscillating_success_count / len(env.data)))
        print('============================')

        if tb_logger:
            tb_logger.add_scalar('epoch/{}/loss'.format(env_name), losses.avg, epoch)
            tb_logger.add_scalar('epoch/{}/dist_from_goal'.format(env_name), dists.avg, epoch)
            tb_logger.add_scalar('epoch/{}/movements'.format(env_name), movements.avg, epoch)
            if self.agent.value_loss is not None:
                tb_logger.add_scalar('epoch/{}/val_loss'.format(env_name), val_losses.avg, epoch)
            if self.agent.val_acc is not None:
                tb_logger.add_scalar('epoch/{}/val_acc'.format(env_name), val_acces.avg, epoch)
            if self.agent.rollback_attn is not None:
                for step in range(self.opts.max_episode_len):
                    tb_logger.add_histogram('epoch_{}/rollback_attn_{}'.format(env_name, step), rollback_attn[step], epoch)
            tb_logger.add_scalar('rollback_oscillation/{}/rollback'.format(env_name), rollback_count / len(env.data), epoch)
            tb_logger.add_scalar('rollback_oscillation/{}/rollback_SR'.format(env_name), rollback_success_count / len(env.data),
                                 epoch)
            tb_logger.add_scalar('rollback_oscillation/{}/oscillating'.format(env_name),
                                 oscillating_count / len(env.data), epoch)
            tb_logger.add_scalar('rollback_oscillation/{}/oscillating_SR'.format(env_name),
                                 oscillating_success_count / len(env.data), epoch)

        # dump into JSON file
        self.agent.results_path = '{}{}_{}_epoch_{}.json'.format(self.opts.results_dir, self.opts.exp_name,
                                                                 env_name, epoch)
        self.agent.write_results()
        score_summary, _ = evaluator.score(self.agent.results_path)
        result_str = ''
        success_rate = 0.0
        for metric, val in score_summary.items():
            result_str += '| {}: {} '.format(metric, val)
            if metric in ['success_rate']:
                success_rate = val
            if tb_logger:
                tb_logger.add_scalar('score/{}/{}'.format(env_name, metric), val, epoch)
        print(result_str)

        return success_rate


def count_rollback_success(success_count, rollback_success_count, rollback_count, oscillating_success_count,
                           oscillating_count, traj, error_margin=3.0):
    """
    check if the viewpoint was visited in the previous 2 steps, if YES, we define this was a rollback action
    We count the total number of successful runs that contain rollback action
    """
    def check_rollback_oscillating(visited_indices, traversed_viewpoints):
        is_rollback = False
        is_oscillating = False
        for index in visited_indices:
            if index + 2 in visited_indices:  # if rollback. Will this condition be too hard?
                is_rollback = True
                if len(traversed_viewpoints) > index + 3:
                    if traversed_viewpoints[index + 1] == traversed_viewpoints[index + 3]:
                        is_oscillating = True

        return (is_rollback, is_oscillating)

    for _traj in traj:
        traversed_viewpoints = [viewpoint for viewpoint, _, _ in _traj['path']]
        is_rollback_oscillating = []
        for viewpoint in traversed_viewpoints:
            visited_indices = [i for i, x in enumerate(traversed_viewpoints) if x == viewpoint]
            is_rollback_oscillating.append(check_rollback_oscillating(visited_indices, traversed_viewpoints))

        is_rollback = False
        is_oscillating = False
        for rollback, oscillating in is_rollback_oscillating:
            if rollback:
                is_rollback = True
            if oscillating:
                is_oscillating = True

        is_success = _traj['distance'][-1] < error_margin
        if is_success:
            success_count += 1
            if is_rollback:
                rollback_success_count += 1
            if is_oscillating:
                oscillating_success_count += 1
        if is_oscillating:
            oscillating_count += 1
        if is_rollback:
            rollback_count += 1

    return success_count, rollback_success_count, rollback_count, oscillating_success_count, oscillating_count