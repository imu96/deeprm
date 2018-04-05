import numpy as np
import math

import job_distribution


class Parameters:
    def __init__(self):

        self.output_filename = 'data/tmp'
        self.test_file = None
        self.num_test_seqs = 1

        self.max_density = 201
        self.min_density = 1.0/201

        self.num_epochs = 10000         # number of training epochs
        self.simu_len = 100       # length of the busy cycle that repeats itself
        self.num_ex = 1                # number of jobsets

        self.output_freq = 1          # interval for output and store parameters

        self.num_seq_per_batch = 10    # number of sequences to compute baseline, actually number of trajectories
        self.episode_max_length = self.simu_len+2  # enforcing an artificial terminal
        self.job_num_cap = self.simu_len          # maximum number of distinct colors in current work graph

        self.num_res = 1               # number of resources in the system
        self.num_nw = 1                # maximum allowed number of work in the queue

        self.time_horizon = self.max_density         # number of time steps in the graph
        self.max_job_len = self.max_density         # maximum duration of new jobs
        self.res_slot = 2200           # maximum number of available resource slots
        self.max_job_size = 1000         # maximum resource request of new work

        self.backlog_size = self.max_density         # backlog queue size

        self.max_track_since_new = 10  # track how many time steps since last new jobs


        self.new_job_rate = 0.7        # lambda in new job arrival Poisson Process
                                	# the job rate would probably be much closer to one
					# e.g. for lambda = 0.7 the expected number of jobs is <3

        self.discount = 1           # discount factor

        # distribution for new job arrival
        self.dist = job_distribution.Dist(self.num_res, self.max_job_size, self.max_job_len)
        # the actual distribution to generate
        self.dist_func = self.dist.uniform_dist

        # graphical representation
        assert self.backlog_size % self.time_horizon == 0  # such that it can be converted into an image
        self.backlog_width = int(math.ceil(self.backlog_size / float(self.time_horizon)))
        self.network_input_height = self.time_horizon
        self.network_input_width = \
            (self.res_slot +
             self.max_job_size * self.num_nw) * self.num_res + \
            self.backlog_width + \
            1  # for extra info, 1) time since last new job

        # compact representation
        self.network_compact_dim = (self.num_res + 1) * \
            (self.time_horizon + self.num_nw) + 1  # + 1 for backlog indicator

        self.network_output_dim = self.num_res+ 1  # + 1 for void action

        self.delay_penalty = -1       # penalty for delaying things in the current work screen
        self.hold_penalty = -1        # penalty for holding things in the new work screen
        self.dismiss_penalty = -1     # penalty for missing a job because the queue is full

        self.num_frames = 1           # number of frames to combine and process
        self.lr_rate = 0.001          # learning rate
        self.rms_rho = 0.9            # for rms prop
        self.rms_eps = 1e-9           # for rms prop

        self.unseen = False  # change random seed to generate unseen example

        # supervised learning mimic policy
        self.batch_size = 10
        self.evaluate_policy_name = "SJF"

    def compute_dependent_parameters(self):
        assert self.backlog_size % self.time_horizon == 0  # such that it can be converted into an image
        self.backlog_width = self.backlog_size / self.time_horizon
        self.network_input_height = self.time_horizon
        self.network_input_width = \
            (self.res_slot +
             self.max_job_size * self.num_nw) * self.num_res + \
            self.backlog_width + \
            1  # for extra info, 1) time since last new job

        # compact representation
        self.network_compact_dim = (self.num_res + 1) * \
            (self.time_horizon + self.num_nw) + 1  # + 1 for backlog indicator

        self.network_output_dim = self.num_res + 1  # + 1 for void action
        self.episode_max_length = self.simu_len+2  # enforcing an artificial terminal
        self.job_num_cap = self.simu_len          # maximum number of distinct colors in current work graph
        self.dist = job_distribution.Dist(self.num_res, self.max_job_size, self.max_job_len)
        # the actual distribution to generate
        self.dist_func = self.dist.uniform_dist

