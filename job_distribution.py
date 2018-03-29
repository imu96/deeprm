import numpy as np
import math


class Dist:

    def __init__(self, num_res, max_job_size, job_len):
        self.num_res = num_res
        self.max_job_size = max_job_size
        self.job_len = job_len

        self.job_small_chance = 0.8

        self.job_len_big_lower = 1
        self.job_len_big_upper = job_len

        self.job_len_small_lower = 1
        self.job_len_small_upper = job_len / 5

        self.dominant_res_lower = max_job_size / 2
        self.dominant_res_upper = max_job_size

        self.other_res_lower = 1
        self.other_res_upper = max_job_size / 5

        self.min_job_len = 1

        self.bad_eta = 1
        self.bad_B = 1
        self.bad_k = int(math.log(job_len / self.min_job_len) / math.log(1 + self.bad_eta))
        self.bad_j = 0
        self.bad_i = 0
        self.bad_w = 1

        self.bimod_w_std = 5
        self.bimod_w_mu = [5, 100]

        self.normal_w_std = 5
        self.normal_w_mu = 10
        self.normal_v_std = 20
        self.normal_v_mu = 50

    # this is the sequence from the online knapsack paper which has the sequence
    # of higher and higher exponential curves
    def bad_dist(self):
        # weight of item is 1
        nw_size = np.zeros(self.num_res)
        for i in range(self.num_res):
            nw_size[i] = self.bad_w

        if self.bad_j > self.bad_k:
            nw_len = 0

        else:
            nw_len = (1 + self.bad_eta)**self.bad_i

        self.bad_i += 1

        if self.bad_i % (self.bad_j + 1) == 0:
            self.bad_i = 0
            self.bad_j += 1

        return nw_len, nw_size

    # after running the distribution, you need to reset the indices so you don't
    # get a bunch of zeroes the next time you ask for the distribution
    def reset_dists(self):
        self.bad_j = 0
        self.bad_i = 0



    # THIS IS A UNIFORM DISTRIBUTION
    def uniform_dist(self):

        # new work duration
        nw_len = np.random.randint(self.min_job_len, self.job_len + 1)  # same length in every dimension

        x = np.random.randint(1, self.max_job_size + 1)
        nw_size = np.ones(self.num_res) * x

#       for i in range(self.num_res):
#           nw_size[i] = np.random.randint(1, self.max_job_size + 1)

        return nw_len, nw_size

    def real_normal_dist(self):
        nw_size = np.zeros(self.num_res)

        for i in range(self.num_res):
            nw_size[i] = np.random.normal(self.normal_w_mu, self.normal_w_std)
            if nw_size[i] < 1:
                nw_size[i] = 1
        #do values
        return nw_len, nw_size

    def bimodal_dist(self):

        #check for going over upper/lower bounds and round off
        #maybe use random.randint to get 1 or 2 and then choose which normal to
       # generate from?
        return


    def const_dist(self):

        nw_len = 4
        nw_size = np.ones(self.num_res) * 5
        return nw_len, nw_size

    def bi_model_dist(self):

        # -- job length --
      #  if np.random.rand() < self.job_small_chance:  # small job
      #      nw_len = np.random.randint(self.job_len_small_lower,
      #                                 self.job_len_small_upper + 1)
      #  else:  # big job
        nw_len = np.random.randint(self.job_len_big_lower,
                                    self.job_len_big_upper + 1)

        nw_size = np.zeros(self.num_res)

        # -- job resource request --
#        dominant_res = np.random.randint(0, self.num_res)
#        for i in range(self.num_res):
#            if i == dominant_res:
        nw_size[0] = np.random.randint(self.dominant_res_lower,
                                           self.dominant_res_upper + 1)
#            else:
#                nw_size[i] = np.random.randint(self.other_res_lower,
#                                               self.other_res_upper + 1)

        return nw_len, nw_size


def generate_sequence_work(pa, seed=42):

    np.random.seed(seed)

    simu_len = pa.simu_len * pa.num_ex

    nw_dist = pa.dist_func

    nw_len_seq = np.zeros(simu_len, dtype=int)
    nw_size_seq = np.zeros((simu_len, pa.num_res), dtype=int)

    for i in range(simu_len):

        #if np.random.rand() < pa.new_job_rate:  # a new job comes
        if (i % pa.simu_len == 0):
            pa.dist.reset_dists()

        nw_len_seq[i], nw_size_seq[i, :] = nw_dist()


    nw_len_seq = np.reshape(nw_len_seq,
                            [pa.num_ex, pa.simu_len])
    nw_size_seq = np.reshape(nw_size_seq,
                             [pa.num_ex, pa.simu_len, pa.num_res])

    return nw_len_seq, nw_size_seq
