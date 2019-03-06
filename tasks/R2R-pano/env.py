import sys

if sys.platform == 'darwin':
    sys.path.append('build_mac')
else:
    sys.path.append('build')
import MatterSim
import csv
import numpy as np
import math
import base64
import random
import networkx as nx

from utils import load_datasets, load_nav_graphs, print_progress, is_experiment

csv.field_size_limit(sys.maxsize)

def load_features(feature_store):
    def _make_id(scanId, viewpointId):
        return scanId + '_' + viewpointId

    # if the tsv file for image features is provided
    if feature_store:
        tsv_fieldnames = ['scanId', 'viewpointId', 'image_w', 'image_h', 'vfov', 'features']
        features = {}
        with open(feature_store, "r") as tsv_in_file:
            print('Reading image features file %s' % feature_store)
            reader = list(csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=tsv_fieldnames))
            total_length = len(reader)

            print('Loading image features ..')
            for i, item in enumerate(reader):
                image_h = int(item['image_h'])
                image_w = int(item['image_w'])
                vfov = int(item['vfov'])
                long_id = _make_id(item['scanId'], item['viewpointId'])
                features[long_id] = np.frombuffer(base64.b64decode(item['features']),
                                                       dtype=np.float32).reshape((36, 2048))
                print_progress(i + 1, total_length, prefix='Progress:',
                               suffix='Complete', bar_length=50)
    else:
        print('Image features not provided')
        features = None
        image_w = 640
        image_h = 480
        vfov = 60
    return features, (image_w, image_h, vfov)


class PanoEnvBatch():
    """ A simple wrapper for a batch of MatterSim environments,
        using discretized viewpoints and pretrained features """

    def __init__(self, features, img_spec, batch_size=64):
        self.features = features
        self.image_w, self.image_h, self.vfov = img_spec

        # initialize list of simulators
        self.sims = []
        for i in range(batch_size):
            sim = MatterSim.Simulator()
            sim.setRenderingEnabled(False)
            sim.setDiscretizedViewingAngles(True)
            sim.setCameraResolution(self.image_w, self.image_h)
            sim.setCameraVFOV(math.radians(self.vfov))
            sim.init()
            self.sims.append(sim)

    def _make_id(self, scanId, viewpointId):
        return scanId + '_' + viewpointId   

    def newEpisodes(self, scanIds, viewpointIds, headings):
        """ Iteratively initialize the simulators for # of batchsize"""
        for i, (scanId, viewpointId, heading) in enumerate(zip(scanIds, viewpointIds, headings)):
            self.sims[i].newEpisode(scanId, viewpointId, heading, 0)

    def getStates(self):
        """ Get list of states augmented with precomputed image features. rgb field will be empty. """
        feature_states = []
        for sim in self.sims:
            state = sim.getState()
            long_id = self._make_id(state.scanId, state.location.viewpointId)
            if self.features:
                feature = self.features[long_id]  # features.shape = (36, 2048)
                feature_states.append((feature, state))
            else:
                feature_states.append((None, state))

        return feature_states


class R2RPanoBatch():
    """ Implements the Room to Room navigation task, using discretized viewpoints and pretrained features """

    def __init__(self, opts, features, img_spec, batch_size=64, seed=10, splits=['train'], tokenizer=None):
        self.env = PanoEnvBatch(features, img_spec, batch_size=batch_size)
        self.data = []
        self.scans = []
        self.opts = opts

        print('Loading {} dataset'.format(splits[0]))

        json_data = load_datasets(splits)
        total_length = len(json_data)

        # iteratively load data into system memory
        for i, item in enumerate(json_data):

            if not is_experiment() and i >= 20: break

            # Split multiple instructions into separate entries
            for j, instr in enumerate(item['instructions']):
                self.scans.append(item['scan'])
                new_item = dict(item)
                new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
                new_item['instructions'] = instr
                if tokenizer:
                    if 'instr_encoding' not in item:  # we may already include 'instr_encoding' when generating synthetic instructions
                        new_item['instr_encoding'] = tokenizer.encode_sentence(instr)
                    else:
                        new_item['instr_encoding'] = item['instr_encoding']
                    if 'divide' in opts.lang_embed:
                        if opts.divide_method == 'kevin':
                            new_item['divid_instr_encoding'] = tokenizer.divide_instr_kevin(instr, opts.max_sentence_segs)
                        else:
                            new_item['divid_instr_encoding'] = tokenizer.divide_instr_victor(instr, opts.max_sentence_segs)
                self.data.append(new_item)
            print_progress(i + 1, total_length, prefix='Progress:',
                               suffix='Complete', bar_length=50)

        self.scans = set(self.scans)
        self.splits = splits
        self.seed = seed
        random.seed(self.seed)
        random.shuffle(self.data)
        self.ix = 0
        self.batch_size = batch_size
        self._load_nav_graphs()
        print('R2RBatch loaded with %d instructions, using splits: %s' % (len(self.data), ",".join(splits)))

    def _load_nav_graphs(self):
        """ Load connectivity graph for each scan, useful for reasoning about shortest paths """
        print('Loading navigation graphs for %d scans' % len(self.scans))
        self.graphs = load_nav_graphs(self.scans)
        self.paths = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self.distances = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def _next_minibatch(self):
        batch = self.data[self.ix:self.ix+self.batch_size]
        if len(batch) < self.batch_size:
            # You ran through all data already, now shuffle data again and take the first few data you need
            random.shuffle(self.data)
            self.ix = self.batch_size - len(batch)
            batch += self.data[:self.ix]
        else:
            self.ix += self.batch_size
        self.batch = batch

    def reset_epoch(self):
        """ Reset the data index to beginning of epoch. Primarily for testing.
            You must still call reset() for a new episode. """
        self.ix = 0

    def _shortest_path_action(self, state, goalViewpointId):
        """ Determine next action on the shortest path to goal, for supervised training. """
        if state.location.viewpointId == goalViewpointId:
            return (0, 0, 0)  # do nothing
        path = self.paths[state.scanId][state.location.viewpointId][goalViewpointId]
        nextViewpointId = path[1]
        # Can we see the next viewpoint?
        for i, loc in enumerate(state.navigableLocations):
            if loc.viewpointId == nextViewpointId:
                # Look directly at the viewpoint before moving
                if loc.rel_heading > math.pi/6.0:
                      return (0, 1, 0)  # Turn right
                elif loc.rel_heading < -math.pi/6.0:
                      return (0,-1, 0)  # Turn left
                elif loc.rel_elevation > math.pi/6.0 and state.viewIndex//12 < 2:
                      return (0, 0, 1)  # Look up
                elif loc.rel_elevation < -math.pi/6.0 and state.viewIndex//12 > 0:
                      return (0, 0,-1)  # Look down
                else:
                      return (i, 0, 0)  # Move
        # Can't see it - first neutralize camera elevation
        if state.viewIndex//12 == 0:
            return (0, 0, 1)  # Look up
        elif state.viewIndex//12 == 2:
            return (0, 0,-1)  # Look down
        # Otherwise decide which way to turn
        target_rel = self.graphs[state.scanId].node[nextViewpointId]['position'] - state.location.point
        target_heading = math.pi/2.0 - math.atan2(target_rel[1], target_rel[0])  # convert to rel to y axis
        if target_heading < 0:
            target_heading += 2.0*math.pi
        if state.heading > target_heading and state.heading - target_heading < math.pi:
            return (0,-1, 0)  # Turn left
        if target_heading > state.heading and target_heading - state.heading > math.pi:
            return (0,-1, 0)  # Turn left
        return (0, 1, 0)  # Turn right

    def _pano_navigable(self, state, goalViewpointId):
        """ Get the navigable viewpoints and their relative heading and elevation,
            as well as the index for 36 image features. """
        navigable_graph = self.graphs[state.scanId].adj[state.location.viewpointId]
        teacher_path = self.paths[state.scanId][state.location.viewpointId][goalViewpointId]

        if len(teacher_path) > 1:
            next_gt_viewpoint = teacher_path[1]
        else:
            # the current viewpoint is our ground-truth
            next_gt_viewpoint = state.location.viewpointId
            gt_viewpoint_idx = (state.location.viewpointId, state.viewIndex)

        # initialize a dict to save info for all navigable points
        # navigable = collections.OrderedDict()
        navigable = {}

        # add the current viewpoint into navigable, so the agent can stay
        navigable[state.location.viewpointId] = {
            'position': state.location.point,
            'heading': state.heading,
            'rel_heading': state.location.rel_heading,
            'rel_elevation': state.location.rel_elevation,
            'index': state.viewIndex
        }

        for viewpoint_id, weight in navigable_graph.items():
            dict_tmp = {}

            node = self.graphs[state.scanId].nodes[viewpoint_id]
            target_rel = node['position'] - state.location.point
            dict_tmp['position'] = list(node['position'])

            # note that this "heading" is computed regarding the global coordinate
            # the actual "heading" between the current viewpoint to next viewpoint
            # needs to take into account the current heading
            target_heading = math.pi / 2.0 - math.atan2(target_rel[1], target_rel[0])  # convert to rel to y axis
            if target_heading < 0:
                target_heading += 2.0*math.pi

            assert state.heading >= 0

            dict_tmp['rel_heading'] = target_heading - state.heading
            dict_tmp['heading'] = target_heading

            # compute the relative elevation
            dist = math.sqrt(sum(target_rel ** 2))  # compute the relative distance
            rel_elevation = np.arcsin(target_rel[2] / dist)
            dict_tmp['rel_elevation'] = rel_elevation

            # elevation level -> 0 (bottom), 1 (middle), 2 (top)
            elevation_level = round(rel_elevation / (30 * math.pi / 180)) + 1
            # To prevent if elevation degree > 45 or < -45
            elevation_level = max(min(2, elevation_level), 0)

            # viewpoint index depends on the elevation as well
            horizontal_idx = int(round(target_heading / (math.pi / 6.0)))
            horizontal_idx = 0 if horizontal_idx == 12 else horizontal_idx
            viewpoint_idx = int(horizontal_idx + 12 * elevation_level)

            dict_tmp['index'] = viewpoint_idx

            # let us get the ground-truth viewpoint index for seq2seq training
            if viewpoint_id == next_gt_viewpoint:
                gt_viewpoint_idx = (viewpoint_id, viewpoint_idx)

            # save into dict
            navigable[viewpoint_id] = dict_tmp

        return navigable, gt_viewpoint_idx

    def heading_elevation_feat(self, state, horizon_views=12, tile=32):
        """ Get heading and elevation features relatively from the current
        heading and elevation """

        assert 360 % horizon_views == 0
        angle = 360 / horizon_views

        rel_heading = np.array(range(0, horizon_views))
        rel_sin_phi = [0] * horizon_views
        rel_cos_phi = [0] * horizon_views
        for i, x in enumerate(rel_heading):
            rel_heading[i] = x * angle - state.heading * 180 / math.pi
            if rel_heading[i] < 0:
                rel_heading[i] = rel_heading[i] + 360

            rel_sin_phi[i] = math.sin(rel_heading[i] / 180 * math.pi)
            rel_cos_phi[i] = math.cos(rel_heading[i] / 180 * math.pi)

        # duplicate the heading features for 3 levels
        rel_sin_phi = np.array(rel_sin_phi * 3)
        rel_cos_phi = np.array(rel_cos_phi * 3)

        rel_elevation = np.array([-1, 0, 1])
        rel_sin_theta = [0] * 3
        rel_cos_theta = [0] * 3
        for i, x in enumerate(rel_elevation):
            rel_elevation[i] = x * angle - state.elevation * 180 / math.pi

            rel_sin_theta[i] = math.sin(rel_elevation[i] / 180 * math.pi)
            rel_cos_theta[i] = math.cos(rel_elevation[i] / 180 * math.pi)
        rel_sin_theta = np.repeat(rel_sin_theta, horizon_views)
        rel_cos_theta = np.repeat(rel_cos_theta, horizon_views)

        feat = np.stack([rel_sin_phi, rel_cos_phi, rel_sin_theta, rel_cos_theta], axis=0)
        feat = np.repeat(feat, tile, axis=0)

        return np.transpose(feat)

    def shortest_path_to_gt_traj(self, state, gt_path):
        """ Compute the next viewpoint by trying to steer back to original ground truth trajectory"""
        min_steps = 100
        min_distance = 100
        current_distance = self.distances[state.scanId][state.location.viewpointId][gt_path[-1]]

        if current_distance != 0:
            for gt_viewpoint in gt_path:
                steps = len(self.paths[state.scanId][state.location.viewpointId][gt_viewpoint])
                next_distance = self.distances[state.scanId][gt_viewpoint][gt_path[-1]]

                # if the next viewpoint requires moving and its distance to the goal is closer
                if steps > 0 and next_distance < current_distance:
                    if min_steps >= steps and min_distance > next_distance:
                        min_steps = steps
                        min_distance = next_distance
                        next_viewpoint = gt_viewpoint
        else:
            next_viewpoint = state.location.viewpointId
        return next_viewpoint

    def _get_obs(self):
        obs = []
        for i, (feature, state) in enumerate(self.env.getStates()):
            item = self.batch[i]

            if self.opts.follow_gt_traj:
                goal_viewpoint = self.shortest_path_to_gt_traj(state, item['path'])
            else:
                goal_viewpoint = item['path'][-1]

            # compute the navigable viewpoints and next ground-truth viewpoint
            navigable, gt_viewpoint_idx = self._pano_navigable(state, goal_viewpoint)

            # get heading and elevation features
            if self.opts.img_fc_use_angle:
                angle_feat = self.heading_elevation_feat(state)
                feature = np.concatenate((feature, angle_feat), axis=1)

            # in synthetic data, path_id is unique since we only has 1 instruction per path, we will then use it as 'instr_id'
            if 'synthetic' in self.splits:
                item['instr_id'] = str(item['path_id'])

            obs.append({
                'instr_id': item['instr_id'],
                'scan': state.scanId,
                'viewpoint': state.location.viewpointId,
                'viewIndex': state.viewIndex,
                'heading': state.heading,
                'elevation': state.elevation,
                'feature': feature,
                'step': state.step,
                'navigableLocations': navigable,
                'instructions': item['instructions'],
                'teacher': item['path'],
                'new_teacher': self.paths[state.scanId][state.location.viewpointId][item['path'][-1]],
                'gt_viewpoint_idx': gt_viewpoint_idx
            })
            if 'instr_encoding' in item:
                obs[-1]['instr_encoding'] = item['instr_encoding']
                obs[-1]['instructions'] = item['instructions']
                if 'divide' in self.opts.lang_embed:
                    obs[-1]['divid_instr_encoding'] = item['divid_instr_encoding']
        return obs

    def reset(self):
        """ Load a new mini-batch / episodes. """
        self._next_minibatch()

        scanIds, viewpointIds, headings = [], [], []
        for item in self.batch:
            scanIds.append(item['scan'])
            viewpointIds.append(item['path'][0])
            headings.append(item['heading'])

        self.env.newEpisodes(scanIds, viewpointIds, headings)
        return self._get_obs()

    def step(self, scanIds, viewpointIds, headings):
        def rotate_to_target_heading(target_heading, state):
            if target_heading < 0:
                target_heading += 2.0 * math.pi
            if abs(target_heading - state.heading) * 180 / math.pi < 15 or abs(target_heading - state.heading) * 180 / math.pi > 345:  # if the target relative heading is less than 15 degree, stop rotating
                return (0, 0, 0)
            if state.heading > target_heading and state.heading - target_heading < math.pi:
                return (0, -1, 0)  # Turn left
            if target_heading > state.heading and target_heading - state.heading > math.pi:
                return (0, -1, 0)  # Turn left
            return (0, 1, 0)  # Turn right

        if self.opts.teleporting:
            self.env.newEpisodes(scanIds, viewpointIds, headings)
        else:
            for i in range(len(self.env.sims)):
                action = None
                # move the agent to the target viewpoint internally, instead of directly 'teleporting'
                while action != (0, 0, 0):
                    state = self.env.sims[i].getState()
                    action = self._shortest_path_action(state, viewpointIds[i])
                    index, heading, elevation = action
                    self.env.sims[i].makeAction(index, heading, elevation)
                # assert state.location.viewpointId == viewpointIds[i], 'the actions took internally was not correct.'

                action = None
                # we have reached the viewpoint, now let's rotate to the corresponding heading
                while action != (0, 0, 0):
                    state = self.env.sims[i].getState()
                    action = rotate_to_target_heading(headings[i], state)
                    index, heading, elevation = action
                    self.env.sims[i].makeAction(index, heading, elevation)
        return self._get_obs()
