import json
import random
import numpy as np

import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F

from utils import padding_idx

class PanoBaseAgent(object):
    """ Base class for an R2R agent with panoramic view and action. """

    def __init__(self, env, results_path):
        self.env = env
        self.results_path = results_path
        random.seed(1)
        self.results = {}
    
    def write_results(self):
        output = []
        for k, v in self.results.items():
            output.append(
                {
                    'instr_id': k,
                    'trajectory': v['path'],
                    'distance': v['distance'],
                    'img_attn': v['img_attn'],
                    'ctx_attn': v['ctx_attn'],
                    'rollback_forward_attn': v['rollback_forward_attn'],
                    'value': v['value'],
                    'viewpoint_idx': v['viewpoint_idx'],
                    'navigable_idx': v['navigable_idx']
                }
            )
        with open(self.results_path, 'w') as f:
            json.dump(output, f)
    
    def _get_distance(self, ob):
        try:
            gt = self.gt[int(ob['instr_id'].split('_')[0])]
        except:  # synthetic data only has 1 instruction per path
            gt = self.gt[int(ob['instr_id'])]
        distance = self.env.distances[ob['scan']][ob['viewpoint']][gt['path'][-1]]
        return distance

    def _select_action(self, logit, ended, is_prob=False, fix_action_ended=True):
        logit_cpu = logit.clone().cpu()
        if is_prob:
            probs = logit_cpu
        else:
            probs = F.softmax(logit_cpu, 1)

        if self.feedback == 'argmax':
            _, action = probs.max(1)  # student forcing - argmax
            action = action.detach()
        elif self.feedback == 'sample':
            # sampling an action from model
            m = D.Categorical(probs)
            action = m.sample()
        else:
            raise ValueError('Invalid feedback option: {}'.format(self.feedback))

        # set action to 0 if already ended
        if fix_action_ended:
            for i, _ended in enumerate(ended):
                if _ended:
                    action[i] = 0

        return action

    def _next_viewpoint(self, obs, viewpoints, navigable_index, action, ended):
        next_viewpoints, next_headings = [], []
        next_viewpoint_idx = []

        for i, ob in enumerate(obs):
            if action[i] >= 1:
                next_viewpoint_idx.append(navigable_index[i][action[i] - 1])  # -1 because the first one in action is 'stop'
            else:
                next_viewpoint_idx.append('STAY')
                ended[i] = True

            # use the available viewpoints and action to select next viewpoint
            next_viewpoints.append(viewpoints[i][action[i]])
            # obtain the heading associated with next viewpoints
            next_headings.append(ob['navigableLocations'][next_viewpoints[i]]['heading'])

        return next_viewpoints, next_headings, next_viewpoint_idx, ended

    def pano_navigable_feat(self, obs, ended):

        # Get the 36 image features for the panoramic view (including top, middle, bottom)
        num_feature, feature_size = obs[0]['feature'].shape

        pano_img_feat = torch.zeros(len(obs), num_feature, feature_size)
        navigable_feat = torch.zeros(len(obs), self.opts.max_navigable, feature_size)

        navigable_feat_index, target_index, viewpoints = [], [], []
        for i, ob in enumerate(obs):
            pano_img_feat[i, :] = torch.from_numpy(ob['feature'])  # pano feature: (batchsize, 36 directions, 2048)

            index_list = []
            viewpoints_tmp = []
            gt_viewpoint_id, viewpoint_idx = ob['gt_viewpoint_idx']

            for j, viewpoint_id in enumerate(ob['navigableLocations']):
                index_list.append(int(ob['navigableLocations'][viewpoint_id]['index']))
                viewpoints_tmp.append(viewpoint_id)

                if viewpoint_id == gt_viewpoint_id:
                    # if it's already ended, we label the target as <ignore>
                    if ended[i] and self.opts.use_ignore_index:
                        target_index.append(self.ignore_index)
                    else:
                        target_index.append(j)

            # we ignore the first index because it's the viewpoint index of the current location
            # not the viewpoint index for one of the navigable directions
            # we will use 0-vector to represent the image feature that leads to "stay"
            navi_index = index_list[1:]
            navigable_feat_index.append(navi_index)
            viewpoints.append(viewpoints_tmp)

            navigable_feat[i, 1:len(navi_index) + 1] = pano_img_feat[i, navi_index]

        return pano_img_feat, navigable_feat, (viewpoints, navigable_feat_index, target_index)

    def pano_navigable_feat_progress_marker(self, step, obs, ended, visited_viewpoints_value,
                                                    progress_marker=1, tiled_len=32, is_training=True):

        # Get the 36 image features for the panoramic view (including top, middle, bottom)
        num_feature, feature_size = obs[0]['feature'].shape

        pano_img_feat = torch.zeros(len(obs), num_feature, feature_size + tiled_len)
        navigable_feat = torch.zeros(len(obs), self.opts.max_navigable, feature_size + tiled_len)

        navigable_feat_index = []
        target_index = []
        viewpoints = []
        visited_index = []
        visited_viewpoints = []
        last_visited_index = []
        previous_oscillation = [0] * len(obs)

        for i, ob in enumerate(obs):
            if tiled_len != 0:
                pano_img_feat[i, :] = torch.cat((torch.from_numpy(ob['feature']),
                                                 torch.ones(num_feature, tiled_len).double()), dim=1)
            else:
                pano_img_feat[i, :] = torch.from_numpy(ob['feature'])

            index_list = []
            viewpoints_tmp = []
            visited_navi_idx_tmp = []
            visited_viewpoints_tmp = []

            gt_viewpoint_id, viewpoint_idx = ob['gt_viewpoint_idx']
            current_viewpoint = ob['viewpoint']

            for j, viewpoint_id in enumerate(ob['navigableLocations']):

                viewpoint_navi_index = int(ob['navigableLocations'][viewpoint_id]['index'])
                index_list.append(viewpoint_navi_index)
                viewpoints_tmp.append(viewpoint_id)

                if viewpoint_id == gt_viewpoint_id:
                    # if it's already ended, we label the target as <ignore>
                    if ended[i] and self.opts.use_ignore_index:
                        target_index.append(self.ignore_index)
                    else:
                        target_index.append(j)

                if step > 0 and viewpoint_id in visited_viewpoints_value[i]:
                    if progress_marker:
                        pano_img_feat[i, viewpoint_navi_index, -tiled_len:].fill_(visited_viewpoints_value[i][viewpoint_id]['value'])

                    visited_navi_idx_tmp.append(j)
                    visited_viewpoints_tmp.append(viewpoint_id)

                    if viewpoint_id == visited_viewpoints_value[i]['latest']:
                        last_visited_index.append(j)

                        # we will block out entirely the possibility of navigating to a viewpoint if that leads to oscillation
                        if current_viewpoint in visited_viewpoints_value[i]:
                            # during training, as long as this is not the target viewpoint for next step
                            if is_training:
                                if 'oscillation_viewpoint' in visited_viewpoints_value[i][current_viewpoint] and viewpoint_id != gt_viewpoint_id:
                                    assert viewpoint_id == visited_viewpoints_value[i][current_viewpoint]['oscillation_viewpoint']  # make sure this is the viewpoint for oscillation
                                    previous_oscillation[i] = 1
                            else:
                                if 'oscillation_viewpoint' in visited_viewpoints_value[i][current_viewpoint]:
                                    assert viewpoint_id == visited_viewpoints_value[i][current_viewpoint]['oscillation_viewpoint']  # make sure this is the viewpoint for oscillation
                                    previous_oscillation[i] = 1


            # we ignore the first index because it's the viewpoint index of the current location
            # not the viewpoint index for one of the navigable directions
            # we will use 0-vector to represent the image feature that leads to "stay"
            navi_index = index_list[1:]
            navigable_feat_index.append(navi_index)
            viewpoints.append(viewpoints_tmp)
            visited_index.append(visited_navi_idx_tmp)
            visited_viewpoints.append(visited_viewpoints_tmp)

            navigable_feat[i, 1:len(navi_index) + 1] = pano_img_feat[i, navi_index]

        if step > 0:
            assert last_visited_index != []
            last_visited_index = [x - 1 for x in last_visited_index]  # for now, -1 is for legacy purpose
        else:
            last_visited_index = [-1] * len(obs)

        return pano_img_feat, navigable_feat, (viewpoints, navigable_feat_index, target_index, visited_index, visited_viewpoints, last_visited_index, previous_oscillation)

    def _sort_batch(self, obs):
        """ Extract instructions from a list of observations and sort by descending
            sequence length (to enable PyTorch packing). """
        seq_tensor = np.array([ob['instr_encoding'] for ob in obs])
        seq_lengths = np.argmax(seq_tensor == padding_idx, axis=1)
        seq_lengths[seq_lengths == 0] = seq_tensor.shape[1]  # Full length
        seq_tensor = torch.from_numpy(seq_tensor)
        return seq_tensor.long().to(self.device), list(seq_lengths)


class PanoSeq2SeqAgent(PanoBaseAgent):
    """ An agent based on an LSTM seq2seq model with attention. """
    def __init__(self, opts, env, results_path, encoder, model, feedback='sample', episode_len=20):
        super(PanoSeq2SeqAgent, self).__init__(env, results_path)
        self.opts = opts
        self.encoder = encoder
        self.model = model
        self.feedback = feedback
        self.episode_len = episode_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.ignore_index = opts.max_navigable + 1  # we define (max_navigable+1) as ignore since 15(navigable) + 1(STOP)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        self.MSELoss = nn.MSELoss()
        self.MSELoss_sum = nn.MSELoss(reduction='sum')

    def get_value_loss_from_start(self, traj, predicted_value, ended):
        """
        This loss forces the agent to estimate how good is the current state, i.e. how far away I am from the goal?
        """
        value_target = []
        for i, _traj in enumerate(traj):
            original_dist = _traj['distance'][0]
            dist = _traj['distance'][-1]
            dist_improved_from_start = (original_dist - dist) / original_dist

            value_target.append(dist_improved_from_start)

            if dist <= 3.0:  # if we are less than 3m away from the goal
                value_target[-1] = 1

            # if ended, let us set the target to be the value so that MSE loss for that sample with be 0
            # we will average the loss according to number of not 'ended', and use reduction='sum' for MSELoss
            if ended[i]:
                value_target[-1] = predicted_value[i].detach()

        value_target = torch.FloatTensor(value_target).to(self.device)

        if self.opts.mse_sum:
            return self.MSELoss_sum(predicted_value.squeeze(), value_target) / sum(1 - ended).item()
        else:
            return self.MSELoss(predicted_value.squeeze(), value_target)

    def get_value_loss_from_start_sigmoid(self, traj, predicted_value, ended):
        """
        This loss forces the agent to estimate how good is the current state, i.e. how far away I am from the goal?
        """
        value_target = []
        for i, _traj in enumerate(traj):
            original_dist = _traj['distance'][0]
            dist = _traj['distance'][-1]
            dist_improved_from_start = (original_dist - dist) / original_dist

            dist_improved_from_start = 0 if dist_improved_from_start < 0 else dist_improved_from_start

            value_target.append(dist_improved_from_start)

            if dist < 3.0:  # if we are less than 3m away from the goal
                value_target[-1] = 1

            # if ended, let us set the target to be the value so that MSE loss for that sample with be 0
            if ended[i]:
                value_target[-1] = predicted_value[i].detach()

        value_target = torch.FloatTensor(value_target).to(self.device)

        if self.opts.mse_sum:
            return self.MSELoss_sum(predicted_value.squeeze(), value_target) / sum(1 - ended).item()
        else:
            return self.MSELoss(predicted_value.squeeze(), value_target)

    def init_traj(self, obs):
        """initialize the trajectory"""
        batch_size = len(obs)

        traj, scan_id = [], []
        for ob in obs:
            traj.append({
                'instr_id': ob['instr_id'],
                'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])],
                'length': 0,
                'feature': [ob['feature']],
                'img_attn': [],
                'ctx_attn': [],
                'rollback_forward_attn': [],
                'value': [],
                'progress_monitor': [],
                'action_confidence': [],
                'regret': [],
                'viewpoint_idx': [],
                'navigable_idx': [],
                'gt_viewpoint_idx': ob['gt_viewpoint_idx'],
                'steps_required': [len(ob['teacher'])],
                'distance': [super(PanoSeq2SeqAgent, self)._get_distance(ob)]
            })
            scan_id.append(ob['scan'])

        self.longest_dist = [traj_tmp['distance'][0] for traj_tmp in traj]
        self.traj_length = [1] * batch_size
        self.value_loss = torch.tensor(0).float().to(self.device)

        ended = np.array([False] * batch_size)
        last_recorded = np.array([False] * batch_size)

        return traj, scan_id, ended, last_recorded

    def update_traj(self, obs, traj, img_attn, ctx_attn, value, next_viewpoint_idx,
                    navigable_index, ended, last_recorded, action_prob=None, rollback=None,
                    rollback_forward_attn=None):
        # Save trajectory output and accumulated reward
        for i, ob in enumerate(obs):
            if not ended[i] or not last_recorded[i]:
                traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))
                dist = super(PanoSeq2SeqAgent, self)._get_distance(ob)
                traj[i]['distance'].append(dist)
                traj[i]['img_attn'].append(img_attn[i].detach().cpu().numpy().tolist())
                traj[i]['ctx_attn'].append(ctx_attn[i].detach().cpu().numpy().tolist())

                if rollback_forward_attn is not None:
                    traj[i]['rollback_forward_attn'].append(rollback_forward_attn[i].detach().cpu().numpy().tolist())

                if len(value[1]) > 1:
                    traj[i]['value'].append(value[i].detach().cpu().tolist())
                else:
                    traj[i]['value'].append(value[i].detach().cpu().item())

                if action_prob is not None:
                    traj[i]['action_confidence'].append(action_prob[i].detach().cpu().item())
                    traj[i]['progress_monitor'].append((action_prob[i] * ((value[i] + 1) / 2)).detach().cpu().item())
                if rollback is not None:
                    traj[i]['regret'].append(rollback[i])

                    if len(traj[i]['regret']) > 1 and traj[i]['regret'][-2] == 1 and traj[i]['regret'][-1] == 1:
                        print('regret twice in the roll: BAD')

                traj[i]['viewpoint_idx'].append(next_viewpoint_idx[i])
                traj[i]['navigable_idx'].append(navigable_index[i])
                traj[i]['steps_required'].append(len(ob['new_teacher']))
                self.traj_length[i] = self.traj_length[i] + 1
                last_recorded[i] = True if ended[i] else False

        return traj, last_recorded

    def rollout_regret(self):
        obs = np.array(self.env.reset())  # load a mini-batch
        batch_size = len(obs)

        seq, seq_lengths = super(PanoSeq2SeqAgent, self)._sort_batch(obs)
        ctx, h_t, c_t, ctx_mask = self.encoder(seq, seq_lengths)
        pre_feat = torch.zeros(batch_size, self.opts.img_feat_input_dim + self.opts.tiled_len).to(self.device)

        # initialize the trajectory
        traj, scan_id, ended, last_recorded = self.init_traj(obs)
        loss = 0
        visited_viewpoints_value = [{} for _ in range(batch_size)]
        value = torch.zeros(batch_size, 1).to(self.device)  # progress monitor values at the beginning
        for step in range(self.opts.max_episode_len):

            pano_img_feat, navigable_feat, viewpoints_indices = \
                super(PanoSeq2SeqAgent, self).pano_navigable_feat_progress_marker(step, obs, ended,
                                                                          visited_viewpoints_value, progress_marker=self.opts.progress_marker,
                                                                          tiled_len=self.opts.tiled_len, is_training=self.encoder.training)
            viewpoints, navigable_index, target_index, visited_navigable_index, visited_viewpoints, navigable_idx_to_previous, previous_oscillation = viewpoints_indices
            block_oscillation = previous_oscillation
            oscillation_index = navigable_idx_to_previous

            pano_img_feat = pano_img_feat.to(self.device)
            navigable_feat = navigable_feat.to(self.device)
            target = torch.LongTensor(target_index).to(self.device)

            # forward pass the network
            h_t, c_t, img_attn, ctx_attn, rollback_forward_attn, \
            logit, rollback_forward_logit, value, navigable_mask = self.model(
                pano_img_feat, navigable_feat, pre_feat, value.detach(), h_t, c_t,
                ctx, navigable_index, navigable_idx_to_previous, oscillation_index, block_oscillation, ctx_mask, seq_lengths,
                prevent_oscillation=self.opts.prevent_oscillation,
                prevent_rollback=self.opts.prevent_rollback,
                is_training=self.encoder.training)

            # Compute the entropy loss of action prob
            # To avoid NaN when multiply prob and logprob, we clone the logit and perform masking
            logit_for_logprob = logit.clone()
            logit.data.masked_fill_((navigable_mask == 0).data, -float('inf'))
            logit_for_logprob.data.masked_fill_((navigable_mask == 0).data, -float('1e7'))

            action_prob = F.softmax(logit, dim=1)
            action_logprob = F.log_softmax(logit_for_logprob, dim=1)
            action_logprob = action_logprob * navigable_mask

            entropy_loss = torch.sum(action_prob * action_logprob, dim=1, keepdim=True).mean()

            # We won't have target to compute loss if it's on test set. In such case, we assign loss 0
            if not self.opts.test_submission:
                if step == 0:
                    current_loss = self.criterion(logit, target) + self.opts.entropy_weight * entropy_loss
                else:
                    if self.opts.monitor_sigmoid:
                        current_val_loss = self.get_value_loss_from_start_sigmoid(traj, value, ended)
                    else:
                        current_val_loss = self.get_value_loss_from_start(traj, value, ended)
                    self.value_loss += current_val_loss

                    current_loss = self.opts.value_loss_weight * current_val_loss + \
                                   (1 - self.opts.value_loss_weight) * self.criterion(logit, target) + \
                                   self.opts.entropy_weight * entropy_loss
            else:
                current_loss = torch.zeros(1)  # during testing where we do not have ground-truth, loss is simply 0
            loss += current_loss

            # select action based on prediction
            action = super(PanoSeq2SeqAgent, self)._select_action(action_prob, ended, is_prob=True)

            next_viewpoints, next_headings, next_viewpoint_idx, ended = super(PanoSeq2SeqAgent, self)._next_viewpoint(
                obs, viewpoints, navigable_index, action, ended)

            # The agent maintain a set of memory M and store the output of the progress monitor associated
            # with each visited viewpoint. It also remembers the last visited viewpoint and
            for minibatch_idx, ob in enumerate(obs):
                current_viewpoint = ob['viewpoint']
                visited_viewpoints_value[minibatch_idx]['latest'] = current_viewpoint

                visited_viewpoints_value[minibatch_idx][current_viewpoint] = {
                    'value': value[minibatch_idx].item()
                }

                # detect if the new viewpoint is a rollback action and make note on which direction will lead to oscillation
                if step > 0:
                    previous_viewpoint = traj[minibatch_idx]['path'][-2][0]
                    next_viewpoint = next_viewpoints[minibatch_idx]
                    if next_viewpoint == previous_viewpoint and not ended[minibatch_idx]:  # rollback
                        visited_viewpoints_value[minibatch_idx][next_viewpoint]['oscillation_viewpoint'] = current_viewpoint

            # option for resetting agent's heading, if it revisited a viewpoint agent
            if step > 0 and self.opts.rollback_reset_heading:
                for minibatch_idx, _traj in enumerate(traj):
                    prev_2_viewpoint = _traj['path'][-2][0]
                    prev_2_heading = _traj['path'][-2][1]
                    if next_viewpoints[minibatch_idx] == prev_2_viewpoint and not ended[minibatch_idx]:
                        next_headings[minibatch_idx] = prev_2_heading

            # make a viewpoint change in the env
            obs = self.env.step(scan_id, next_viewpoints, next_headings)

            pre_feat = navigable_feat[torch.LongTensor(range(batch_size)), action, :]

            # Save trajectory output and accumulated reward
            traj, last_recorded = self.update_traj(obs, traj, img_attn, ctx_attn, value, next_viewpoint_idx,
                                                   navigable_index, ended, last_recorded,
                                                   rollback_forward_attn=rollback_forward_attn)

            # Early exit if all ended
            if last_recorded.all():
                break

        self.dist_from_goal = [traj_tmp['distance'][-1] for traj_tmp in traj]

        return loss, traj

    def rollout_monitor(self):
        obs = np.array(self.env.reset())  # load a mini-batch
        batch_size = len(obs)

        seq, seq_lengths = super(PanoSeq2SeqAgent, self)._sort_batch(obs)

        ctx, h_t, c_t, ctx_mask = self.encoder(seq, seq_lengths)
        question = h_t

        if self.opts.arch == 'progress_aware_marker' or self.opts.arch == 'iclr_marker':
            pre_feat = torch.zeros(batch_size, self.opts.img_feat_input_dim + self.opts.tiled_len).to(self.device)
        else:
            pre_feat = torch.zeros(batch_size, self.opts.img_feat_input_dim).to(self.device)

        # Mean-Pooling over segments as previously attended ctx
        pre_ctx_attend = torch.zeros(batch_size, self.opts.rnn_hidden_size).to(self.device)

        # initialize the trajectory
        traj, scan_id, ended, last_recorded = self.init_traj(obs)

        loss = 0
        for step in range(self.opts.max_episode_len):

            pano_img_feat, navigable_feat, \
            viewpoints_indices = super(PanoSeq2SeqAgent, self).pano_navigable_feat(obs, ended)
            viewpoints, navigable_index, target_index = viewpoints_indices

            pano_img_feat = pano_img_feat.to(self.device)
            navigable_feat = navigable_feat.to(self.device)
            target = torch.LongTensor(target_index).to(self.device)

            # forward pass the network
            h_t, c_t, pre_ctx_attend, img_attn, ctx_attn, logit, value, navigable_mask = self.model(
                pano_img_feat, navigable_feat, pre_feat, question, h_t, c_t, ctx,
                pre_ctx_attend, navigable_index, ctx_mask)

            # set other values to -inf so that logsoftmax will not affect the final computed loss
            logit.data.masked_fill_((navigable_mask == 0).data, -float('inf'))
            current_logit_loss = self.criterion(logit, target)
            # select action based on prediction
            action = super(PanoSeq2SeqAgent, self)._select_action(logit, ended, fix_action_ended=self.opts.fix_action_ended)

            if not self.opts.test_submission:
                if step == 0:
                    current_loss = current_logit_loss
                else:
                    if self.opts.monitor_sigmoid:
                        current_val_loss = self.get_value_loss_from_start_sigmoid(traj, value, ended)
                    else:
                        current_val_loss = self.get_value_loss_from_start(traj, value, ended)

                    self.value_loss += current_val_loss
                    current_loss = self.opts.value_loss_weight * current_val_loss + (
                            1 - self.opts.value_loss_weight) * current_logit_loss
            else:
                current_loss = torch.zeros(1)  # during testing where we do not have ground-truth, loss is simply 0
            loss += current_loss

            next_viewpoints, next_headings, next_viewpoint_idx, ended = super(PanoSeq2SeqAgent, self)._next_viewpoint(
                obs, viewpoints, navigable_index, action, ended)

            # make a viewpoint change in the env
            obs = self.env.step(scan_id, next_viewpoints, next_headings)

            # save trajectory output and update last_recorded
            traj, last_recorded = self.update_traj(obs, traj, img_attn, ctx_attn, value, next_viewpoint_idx,
                                                   navigable_index, ended, last_recorded)

            pre_feat = navigable_feat[torch.LongTensor(range(batch_size)), action,:]

            # Early exit if all ended
            if last_recorded.all():
                break

        self.dist_from_goal = [traj_tmp['distance'][-1] for traj_tmp in traj]

        return loss, traj

    def rollout(self):
        obs = np.array(self.env.reset())  # load a mini-batch
        batch_size = len(obs)

        seq, seq_lengths = super(PanoSeq2SeqAgent, self)._sort_batch(obs)
        ctx, h_t, c_t, ctx_mask = self.encoder(seq, seq_lengths)

        pre_feat = torch.zeros(batch_size, obs[0]['feature'].shape[1]).to(self.device)

        # initialize the trajectory
        traj, scan_id = [], []
        for ob in obs:
            traj.append({
                'instr_id': ob['instr_id'],
                'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])],
                'length': 0,
                'feature': [ob['feature']],
                'ctx_attn': [],
                'value': [],
                'viewpoint_idx': [],
                'navigable_idx': [],
                'gt_viewpoint_idx': ob['gt_viewpoint_idx'],
                'steps_required': [len(ob['teacher'])],
                'distance': [super(PanoSeq2SeqAgent, self)._get_distance(ob)]
            })
            scan_id.append(ob['scan'])

        self.longest_dist = [traj_tmp['distance'][0] for traj_tmp in traj]
        self.traj_length = [1] * batch_size
        ended = np.array([False] * len(obs))
        last_recorded = np.array([False] * len(obs))
        loss = 0
        for step in range(self.opts.max_episode_len):

            pano_img_feat, navigable_feat, \
            viewpoints_indices = super(PanoSeq2SeqAgent, self).pano_navigable_feat(obs, ended)
            viewpoints, navigable_index, target_index = viewpoints_indices
            pano_img_feat = pano_img_feat.to(self.device)
            navigable_feat = navigable_feat.to(self.device)

            # get target
            target = torch.LongTensor(target_index).to(self.device)

            # forward pass the network
            h_t, c_t, ctx_attn, logit, navigable_mask = self.model(pano_img_feat, navigable_feat, pre_feat, h_t, c_t, ctx, navigable_index, ctx_mask)

            # we mask out output
            logit.data.masked_fill_((navigable_mask == 0).data, -float('inf'))

            loss += self.criterion(logit, target)

            # select action based on prediction
            action = super(PanoSeq2SeqAgent, self)._select_action(logit, ended)
            next_viewpoints, next_headings, next_viewpoint_idx, ended = super(PanoSeq2SeqAgent, self)._next_viewpoint(
                obs, viewpoints, navigable_index, action, ended)

            # make a viewpoint change in the env
            obs = self.env.step(scan_id, next_viewpoints, next_headings)

            # Save trajectory output and update last_recorded
            for i, ob in enumerate(obs):
                if not ended[i] or not last_recorded[i]:
                    traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))
                    dist = super(PanoSeq2SeqAgent, self)._get_distance(ob)
                    traj[i]['distance'].append(dist)
                    traj[i]['ctx_attn'].append(ctx_attn[i].detach().cpu().numpy().tolist())
                    traj[i]['viewpoint_idx'].append(next_viewpoint_idx[i])
                    traj[i]['navigable_idx'].append(navigable_index[i])
                    traj[i]['steps_required'].append(len(ob['new_teacher']))
                    self.traj_length[i] = self.traj_length[i] + 1
                    last_recorded[i] = True if ended[i] else False

            pre_feat = navigable_feat[torch.LongTensor(range(batch_size)), action,:]

            # Early exit if all ended
            if last_recorded.all():
                break

        self.dist_from_goal = [traj_tmp['distance'][-1] for traj_tmp in traj]

        return loss, traj
