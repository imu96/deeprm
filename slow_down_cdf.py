import sys
import os
import numpy as np
import scipy.stats as stats
import cPickle
import matplotlib.pyplot as plt

import environment
import parameters
import pg_network
import other_agents

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

def discount(x, gamma):
    """
    Given vector x, computes a vector y such that
    y[i] = x[i] + gamma * x[i+1] + gamma^2 x[i+2] + ...
    """
    out = np.zeros(len(x))
    out[-1] = x[-1]
    for i in reversed(xrange(len(x)-1)):
        out[i] = x[i] + gamma*out[i+1]
    assert x.ndim >= 1
    # More efficient version:
    # scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]
    return out


def categorical_sample(prob_n):
    """
    Sample from categorical distribution,
    specified by a vector of class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np.random.rand()).argmax()


def get_traj(test_type, pa, env, episode_max_length, pg_resume=None, render=False, upper=None, lower=None, cap=None):
    """
    Run agent-environment loop for one whole episode (trajectory)
    Return dictionary of results
    """

    if test_type == 'PG':  # load trained parameters

        pg_learner = pg_network.PGLearner(pa)

        net_handle = open(pg_resume, 'rb')
        net_params = cPickle.load(net_handle)
        pg_learner.set_net_params(net_params)

    env.reset()
    if cap is not None:
        cap_diff = pa.res_slot - cap
        env.machine.canvas[:, 0, : cap_diff] = 1
        env.machine.avbl_slot[:,:] = cap

    rews = []
    ob = env.observe()

    for _ in xrange(episode_max_length):

        if test_type == 'PG':
            a = pg_learner.choose_action(ob)

        elif test_type == 'KP':
            if pa.test_file is not None:
                a = other_agents.get_kp_action(env.machine, env.job_slot, upper, lower, cap=cap)
            else:
                a = other_agents.get_kp_action(env.machine, env.job_slot, pa.max_density, pa.min_density, cap=cap)

        elif test_type == 'Tetris':
            a = other_agents.get_packer_action(env.machine, env.job_slot)

        elif test_type == 'SJF':
            a = other_agents.get_sjf_action(env.machine, env.job_slot)

        elif test_type == 'Random':
            a = other_agents.get_random_action(pa.num_res)

        ob, rew, done, info = env.step(a, repeat=True, test_type=test_type)

        rews.append(rew)

        if done: break
        if render: env.render()
        # env.render()

    if pa.test_file is not None:
        rews -= (pa.res_slot - cap)

    return np.array(rews), info

def read_test_file(test_dir, num_seqs=1):
    last_dir = test_dir.split("/")[-1]
    test_caps_file = test_dir + "/caps_" + last_dir
    test_caps = np.genfromtxt(test_caps_file, dtype=None)

    optim_file = test_dir + "/opts_" + last_dir
    optim = np.genfromtxt(optim_file, dtype=None)

    num_jobs = int(last_dir.split("_")[1])
    num_exs = len(test_caps)

    test_len_seqs = np.zeros((num_exs*num_seqs, num_jobs), dtype=int)
    test_size_seqs = np.zeros((num_exs*num_seqs, num_jobs, 1), dtype=int)
    perms = []

    test_file_prefix = test_dir + "/" + last_dir + "_"
    seq_idx = 0
    for i in range(0, num_exs):
        if i < 10:
            test_file = test_file_prefix + "0" + str(i)
        else:
            test_file = test_file_prefix + str(i)
        values, weights = np.genfromtxt(test_file, delimiter=",",
                                            usecols=(1,2), unpack=True, dtype=None)
        density = np.divide(values, weights, dtype=float)
        if max(density) > 201 or min(density) < 1./201:
            print max(density), min(density)
            print "Example " + str(i) + " cannot be tested"
            continue
        for j in range(num_seqs):
            perm = np.random.permutation(num_jobs)
            perms.append(perm)
            test_len_seqs[seq_idx,:] = values[perm]
            test_size_seqs[seq_idx,:,:] = np.reshape(weights[perm], (num_jobs, 1))
            seq_idx += 1

    return test_len_seqs, test_size_seqs, test_caps, optim, perms

def launch(pa, pg_resume=None, render=False, plot=False, repre='image', end='no_new_job', cap=None):

    # ---- Parameters ----

    test_types = ['KP', 'Random']

    ratio_comparers = {
        'optim_type': 'KP',   #denominator
        'learned_type': 'PG'  #numerator
    }

    test_dists = {
        "1": "Uncorrelated",
        "2": "Weakly correlated",
        "3": "Strongly correlated",
        "4": "Inverse strongly correlated",
        "5": "Almost strongly correlated",
        "6": "Subset sum",
        "11": "Uncorrelated Span(2,10)",
        "12": "Weakly correlated Span(2,10)",
        "13": "Strongly correlated Span(2,10)",
        "14": "Multiple strongly correlated (300, 200, 6)",
        "15": "Profit Ceiling (3)",
        "16": "Circle (2/3)"
    }

    test_dists_LU = {
        "1": [1./1000, 1000],
        "2": [1./101, 101],
        "3": [1.1, 101],
        "4": [1./101, 10./11],
        "5": [549./500, 103],
        "6": [1, 1],
        "11": [1./200, 200],
        "12": [1./21, 21],
        "13": [1.1, 21],
        "14": [1.2, 201],
        "15": [1,3],
        "16": [1,43]
    }

    epsilon = 1

    if pg_resume is not None:
        test_types = ['PG'] + test_types

    if pa.test_file is not None:
        test_len_seqs, test_size_seqs, test_caps, off_optim, perms = read_test_file(pa.test_file, pa.num_test_seqs)
        pa.num_ex = len(test_caps)
        last_dir = pa.test_file.split("/")[-1]
        dist = last_dir.split("_")[0]
        n = int(last_dir.split("_")[1])
        pa.simu_len = n
        pa.compute_dependent_parameters()
        env = environment.Env(pa, nw_len_seqs=test_len_seqs, nw_size_seqs=test_size_seqs,
                                render=render, repre=repre, end=end)

        counter = 0
        perms_file = pa.output_filename + "_perms.txt"
        with open(perms_file, 'w') as f:
            f.write("Distribution " + dist + " Sequences\n\n")
            for i in xrange(pa.num_ex):
                f.write("Example "+str(i)+":\n")
                for j in xrange(pa.num_test_seqs):
                    f.write(str(perms[counter])+"\n")
                    counter += 1
                f.write("\n")

    else:
        env = environment.Env(pa, render=render, repre=repre, end=end)
        nw_len_seqs = env.nw_len_seqs
        nw_size_seqs = env.nw_size_seqs

        item_file = pa.output_filename + '_items.txt'
        with open(item_file, 'w') as f:
            f.write("Format: Job # (size, value)\n\n")
            for j in xrange(0, len(nw_len_seqs)):
                f.write("Jobset "+str(j)+"\n")
                for i in xrange(0,len(nw_len_seqs[j])):
                    job_str = "Job "+str(i)+":"+"\t"+str(nw_size_seqs[j][i][0])+"\t"+str(nw_len_seqs[j][i]) + "\n"
                    f.write(job_str)
                f.write("\n")

    all_discount_rews = {}
    all_knapsac_vals = {}
    disc_rew_ratios = {}
    knapsac_val_ratios = {}

    for test_type in test_types:

        all_discount_rews[test_type] = []
        all_knapsac_vals[test_type] = []
        disc_rew_ratios[test_type] = []
        knapsac_val_ratios[test_type] = []

    test_dist_func = pa.dist_func.__name__
    dist_num = test_dist_func.split("_")[-1]
    pisinger_dist = dist_num != "dist"

    for seq_idx in xrange(pa.num_ex):
        print("\n=============== " + str(seq_idx) + " ===============")

        start = env.seq_no
        for test_type in test_types:
            env.seq_no = start
            print "---------- " + test_type + " -----------"
            if pa.test_file is not None:
                LU = test_dists_LU[dist]
                print "optimal offline value in knapsack : \t %s" % (off_optim[seq_idx])

            for i in xrange(pa.num_test_seqs):
                blockPrint()
                test_dist_func
                if pa.test_file is not None:
                    rews, info = get_traj(test_type, pa, env, pa.episode_max_length, pg_resume,
                                            upper=LU[1], lower=LU[0], cap=test_caps[seq_idx])
                elif pisinger_dist:
                    LU = test_dists_LU[dist_num]
                    rews, info = get_traj(test_type, pa, env, pa.episode_max_length, pg_resume,
                                            upper=LU[1], lower=LU[0], cap=cap)
                else:
                    rews, info = get_traj(test_type, pa, env, pa.episode_max_length, pg_resume, cap=cap)
                enablePrint()

                if pa.num_test_seqs > 1:
                    print "value in knapsack (seq %s): \t %s" % (i+1, rews[-1])
                else:
                    print "value in knapsack : \t %s" % (rews[-1])

                all_discount_rews[test_type].append(
                    discount(rews, pa.discount)[0]
                )

                all_knapsac_vals[test_type].append(
                    float(rews[-1])
                )
                env.seq_no += 1
        env.seq_no = start + pa.num_test_seqs


    # let's crunch some statistics!

    knapsac_off_ratios = {}
    ex_val_ratios = {}
    max_ex_val_ratios = {}

    for test_type in test_types:
        disc_rew_ratios[test_type] = np.divide(all_discount_rews[test_type], all_discount_rews[ratio_comparers['optim_type']])
        knapsac_val_ratios[test_type] = np.divide(all_knapsac_vals[test_type], all_knapsac_vals[ratio_comparers['optim_type']])
        if pa.test_file is not None:
            knapsac_off_ratios[test_type] = np.divide(all_knapsac_vals[test_type], np.repeat(off_optim, pa.num_test_seqs))


    if pg_resume is not None:
        if pa.test_file is not None:
            print "\nTesting Results for Distribution " + dist + ": " + test_dists[dist]
        elif pisinger_dist:
            print "\nTesting Results for Distribution " + dist_num + ": " + test_dists[dist_num]

        print "\nStatistics: "

        for test_type in test_types:

            if pa.test_file is not None:

                max_off_ratio = np.max(knapsac_off_ratios[test_type])
                min_off_ratio = np.min(knapsac_off_ratios[test_type])
                avg_off_ratio = np.mean(knapsac_off_ratios[test_type])
                geo_adj_avg_off_ratio = stats.gmean(knapsac_off_ratios[test_type]+1)
                std_off_ratio = np.std(knapsac_off_ratios[test_type])

                print "\n**Performance of " + test_type + " against Offline Knapsack Solution" + ":\n"

                print "Max Knapsack Value Ratio: \t\t" + str(max_off_ratio)
                print "Min Knapsack Value Ratio: \t\t" + str(min_off_ratio)
                print "Average of Knapsack Value Ratios: \t" + str(avg_off_ratio)
                print "Geometric Mean (Adj) of Knapsack Value Ratios: \t" + str(geo_adj_avg_off_ratio)
                print "Standard Deviation of Knapsack Value Ratios: \t +-" + str(std_off_ratio)

                if pa.num_test_seqs > 1:
                    ex_val_ratios[test_type] = knapsac_off_ratios[test_type].reshape(pa.num_ex, pa.num_test_seqs)
                    max_ex_val_ratios[test_type] = np.array([max(r) for r in ex_val_ratios[test_type]])
                    mean_max_off = np.mean(max_ex_val_ratios[test_type])
                    mean_min_off = np.mean([min(r) for r in ex_val_ratios[test_type]])
                    mean_std_off = np.mean([np.std(r) for r in ex_val_ratios[test_type]])

                    gmean_max_off = stats.gmean(max_ex_val_ratios[test_type]+1)
                    gmean_min_off = stats.gmean([min(r)+1 for r in ex_val_ratios[test_type]])
                    gmean_std_off = stats.gmean([np.std(r)+1 for r in ex_val_ratios[test_type]])

                    print "\nWithin Test Instances Performance:"
                    print "Mean of Max Knapsack Value Ratio: \t\t" + str(mean_max_off)
                    print "Mean of Min Knapsack Value Ratio: \t\t" + str(mean_min_off)
                    print "Mean of Standard Deviation of Knapsack Value Ratios: \t +-" + str(mean_std_off)
                    print "Geo Mean of Max Knapsack Value Ratio: \t\t" + str(gmean_max_off)
                    print "Geo Mean of Min Knapsack Value Ratio: \t\t" + str(gmean_min_off)
                    print "Geo Mean of Standard Deviation of Knapsack Value Ratios: \t +-" + str(gmean_std_off)


            if test_type == ratio_comparers["optim_type"]:
                continue

            max_knap_val = np.max(knapsac_val_ratios[test_type])
            min_knap_val = np.min(knapsac_val_ratios[test_type])
            avg_knap_val = np.mean(knapsac_val_ratios[test_type])
            geo_avg_knap_val = stats.gmean(knapsac_val_ratios[test_type]+1)
            std_knap_val = np.std(knapsac_val_ratios[test_type])

            print "\n**Performance of " + test_type + " against " + ratio_comparers["optim_type"] + ":\n"

            print "Max Knapsack Value Ratio: \t\t" + str(max_knap_val)
            print "Min Knapsack Value Ratio: \t\t" + str(min_knap_val)
            print "Average of Knapsack Value Ratios: \t" + str(avg_knap_val)
            print "Geometric Mean (adj) of Knapsack Value Ratios: \t" + str(geo_avg_knap_val)
            print "Standard Deviation of Knapsack Value Ratios: \t +-" + str(std_knap_val)

    # -- matplotlib colormap no overlap --
    if plot:
        num_colors = len(test_types)

        cm = plt.get_cmap('gist_rainbow')
        fig = plt.figure(figsize=(12, 5))
        if pa.test_file is not None:
            ax = fig.add_subplot(121)
        else:
            ax = fig.add_subplot(111)
        ax.set_prop_cycle('color', [cm(1. * i / num_colors) for i in range(num_colors)])

        for test_type in test_types:
            val_perf_cdf = np.sort(knapsac_val_ratios[test_type])
            val_perf_yvals = np.arange(len(val_perf_cdf))/float(len(val_perf_cdf) - 1)
            ax.plot(val_perf_cdf, val_perf_yvals, label=test_type)

        ax.legend(loc=0)
        ax.set_xlabel("Performance vs. KP")
        ax.set_ylabel("CDF")

        if pa.test_file is not None:
            fig.suptitle("Distribution "+dist+": "+test_dists[dist]+"\n"+str(pa.num_test_seqs) + " Input Sequence(s)")
        elif pisinger_dist:
            fig.suptitle("Distribution "+dist_num+": "+test_dists[dist_num])

        if pa.test_file is not None:
            if pa.num_test_seqs > 1:
                num_colors += len(test_types)
            cm = plt.get_cmap('gist_rainbow')
            ax2 = fig.add_subplot(122)
            ax2.set_prop_cycle('color', [cm(1. * i / num_colors) for i in range(num_colors)])

            for test_type in test_types:
                val_perf_cdf = np.sort(knapsac_off_ratios[test_type])
                val_perf_yvals = np.arange(len(val_perf_cdf))/float(len(val_perf_cdf) - 1)
                ax2.plot(val_perf_cdf, val_perf_yvals, label=test_type)

                if pa.num_test_seqs > 1:
                    mval_perf_cdf = np.sort(max_ex_val_ratios[test_type])
                    mval_perf_yvals = np.arange(len(mval_perf_cdf))/float(len(mval_perf_cdf) - 1)
                    ax2.plot(mval_perf_cdf, mval_perf_yvals, label=test_type+" Max")

            ax2.legend(loc=0)
            ax2.set_xlabel("Performance vs. Offline Solution")
            ax2.set_ylabel("CDF")

        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        plt.savefig(pa.output_filename + "_ratio_fig" + ".pdf")
        plt.close()

    return all_discount_rews, all_knapsac_vals

def main():
    pa = parameters.Parameters()

    pa.simu_len = 200  # 5000  # 1000
    pa.num_ex = 10  # 100
    pa.num_nw = 10
    pa.num_seq_per_batch = 20
    # pa.max_nw_size = 5
    # pa.job_len = 5
    pa.new_job_rate = 0.3
    pa.discount = 1

    pa.episode_max_length = 20000  # 2000

    pa.compute_dependent_parameters()

    render = False

    plot = True  # plot slowdown cdf

    pg_resume = None
    pg_resume = 'data/pg_re_discount_1_rate_0.3_simu_len_200_num_seq_per_batch_20_ex_10_nw_10_1450.pkl'
    # pg_resume = 'data/pg_re_1000_discount_1_5990.pkl'

    pa.unseen = True

    launch(pa, pg_resume, render, plot, repre='image', end='all_done')


if __name__ == '__main__':
    main()
