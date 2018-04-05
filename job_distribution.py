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

        # For spanning distributions
        self.uncorr_span = []
        self.weak_span = []
        self.strong_span = []

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
    # create new spans for each instance (items set) for spanning distributions
    def reset_dists(self):
        self.bad_j = 0
        self.bad_i = 0
        self.create_span()

    # THIS IS A UNIFORM DISTRIBUTION
    def uniform_dist(self):

        # REMEMBER: size == weight & len == value

        x = np.random.randint(1, self.max_job_size + 1)
        nw_size = np.ones(self.num_res) * x
        nw_len = np.random.randint(1, self.job_len*nw_size[0] + 1)

        return nw_len, nw_size

    def create_span(self):
        w1 = np.random.randint(1, self.max_job_size+1)
        w2 = np.random.randint(1, self.max_job_size+1)

        p1 = np.random.randint(1, self.max_job_size+1)
        p2 = np.random.randint(1, self.max_job_size+1)
        self.uncorr_span = [{"w": w1, "p": p1}, {"w": w2, "p": p2}]

        m1 = max(1, w1 - 100)
        p1 = np.random.randint(m1, w1+101)
        m2 = max(1, w2 - 100)
        p2 = np.random.randint(m2, w2+101)
        self.uncorr_span = [{"w": w1, "p": p1}, {"w": w2, "p": p2}]

        p1 = w1 + 100
        p2 = w2 + 100
        self.uncorr_span = [{"w": w1, "p": p1}, {"w": w2, "p": p2}]

    def weak_dist_2(self):
        """
        Weakly correlated distribution
        w: [1,1000]
        p: [w-100, w+100], p>= 1
        """

        w = np.random.randint(1, self.max_job_size + 1)
        nw_size = np.ones(self.num_res) * w

        m = np.max([1, w - 100])
        nw_len = np.random.randint(m, w + 101)

        return nw_len, nw_size

    def strong_dist_3(self):
        """
        Strongly correlated distribution
        w: [1,1000]
        p: w+100
        """

        w = np.random.randint(1, self.max_job_size+1)
        nw_size = np.ones(self.num_res) * w
        nw_len = w + 100

        return nw_len, nw_size

    def inverse_strong_dist_4(self):
        """
        Inverse strongly correlated distribution
        w: p + 100
        p: [1, 1000]
        """

        nw_len = np.random.randint(1, self.max_job_size+1)
        w = nw_len + 100
        nw_size = np.ones(self.num_res) * w

        return nw_len, nw_size

    def almost_strong_dist_5(self):
        """
        Almost strongly correlated distribution
        w: [1,1000]
        p: [w+100-2, w+100+2]
        """

        w = np.random.randint(1, self.max_job_size+1)
        nw_size = np.ones(self.num_res) * w
        nw_len = np.random.randint(w+98, w+102+1)

        return nw_len, nw_size

    def subset_sum_dist_6(self):
        """
        Subset sum distribution
        w: [1, 1000]
        p: w
        """

        w = np.random.randint(1, self.max_job_size+1)
        nw_size = np.ones(self.num_res) * w
        nw_len = w

        return nw_len, nw_size

    def uncorr_span_dist_11(self):
        """
        Uncorrelated span (2, 10) distribution
        """

        which_vec = np.random.randint(0, 2)
        a = np.random.randint(1, 11)

        nw_size = np.ones(self.num_res) * a * uncorr_span[which_vec]["w"]
        nw_len = a * uncorr_span[which_vec]["p"]

        return nw_len, nw_size

    def weak_span_dist_12(self):
        """
        Weakly correlated span (2, 10) distribution
        """

        which_vec = np.random.randint(0, 2)
        a = np.random.randint(1, 11)

        nw_size = np.ones(self.num_res) * a * weak_span[which_vec]["w"]
        nw_len = a * weak_span[which_vec]["p"]

        return nw_len, nw_size

    def uncorr_span_dist_13(self):
        """
        Strongly correlated span (2, 10) distribution
        """

        which_vec = np.random.randint(0, 2)
        a = np.random.randint(1, 11)

        nw_size = np.ones(self.num_res) * a * strong_span[which_vec]["w"]
        nw_len = a * strong_span[which_vec]["p"]

        return nw_len, nw_size

    def mult_strong_dist_14(self):
        """
        Multiple strongly correlated distribution (300, 200, 6)
        w: [1, 1000]
        if 6 | w : p = w + 300
        else: p = w + 200
        """

        w = np.random.randint(1, self.max_job_size+1)
        nw_size = np.ones(self.num_res) * w

        if w % 6 == 0:
            nw_len = w + 300
        else:
            nw_len = w + 200

        return nw_len, nw_size

    def profit_ceil_dist_15(self):
        """
        Profit ceiling distribution (3)
        w: [1, 1000]
        p: 3*ceiling(w/3)
        """

        w = np.random.randint(1, self.max_job_size+1)
        nw_size = np.ones(self.num_res) * w
        p = 3 * np.ceil(w/3.)
        nw_len = int(p)

        return nw_len, nw_size

    def circle_dist_16(self):
        """
        Circle distribution (2/3)
        w: [1, 1000]
        p: (2/3)*sqrt(4*1000^2 - (w - 2000)^2)
        """

        w = np.random.randint(1, self.max_job_size+1)
        nw_size = np.ones(self.num_res) * w
        p = (2./3.) * np.sqrt(4*1000^2 - (w - 2000)^2)
        nw_len = int(round(p))

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
