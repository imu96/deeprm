import numpy as np
import cPickle
import matplotlib.pyplot as plt

import environment
import parameters
import pg_network
import other_agents


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


def get_traj(test_type, pa, env, episode_max_length, pg_resume=None, render=False):
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
    rews = []

    ob = env.observe()

    for _ in xrange(episode_max_length):

        if test_type == 'PG':
            a = pg_learner.choose_action(ob)

        elif test_type == 'KP':
            a = other_agents.get_kp_action(env.machine, env.job_slot, pa.max_density, pa.min_density)

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

    return np.array(rews), info


def launch(pa, pg_resume=None, render=False, plot=False, repre='image', end='no_new_job'):

    # ---- Parameters ----

    test_types = ['KP', 'Random']

    ratio_comparers = {
        'optim_type': 'KP',   #denominator
        'learned_type': 'PG'  #numerator
    }

    epsilon = 1

    if pg_resume is not None:
        test_types = ['PG'] + test_types

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

    jobs_slow_down = {}
    work_complete = {}
    work_remain = {}
    job_len_remain = {}
    num_job_remain = {}
    job_remain_delay = {}

    for test_type in test_types:

        all_discount_rews[test_type] = []
        all_knapsac_vals[test_type] = []
        disc_rew_ratios[test_type] = []
        knapsac_val_ratios[test_type] = []

        jobs_slow_down[test_type] = []
        work_complete[test_type] = []
        work_remain[test_type] = []
        job_len_remain[test_type] = []
        num_job_remain[test_type] = []
        job_remain_delay[test_type] = []

   # for seq_idx in xrange(10):
    for seq_idx in xrange(pa.num_ex):
        print("\n=============== " + str(seq_idx) + " ===============")

        for test_type in test_types:

            rews, info = get_traj(test_type, pa, env, pa.episode_max_length, pg_resume)

            print "---------- " + test_type + " -----------"

            print "total discount reward : \t %s" % (discount(rews, pa.discount)[0])
            print "value in knapsack : \t %s" % (rews[-1])

            all_discount_rews[test_type].append(
                discount(rews, pa.discount)[0]
            )

            all_knapsac_vals[test_type].append(
                float(rews[-1])
            )

            # ------------------------
            # ---- per job stat ----
            # ------------------------

            enter_time = np.array([info.record[i].enter_time for i in xrange(len(info.record))])
            finish_time = np.array([info.record[i].finish_time for i in xrange(len(info.record))])
            job_len = np.array([info.record[i].len for i in xrange(len(info.record))])
            job_total_size = np.array([np.sum(info.record[i].res_vec) for i in xrange(len(info.record))])

            finished_idx = (finish_time >= 0)
            unfinished_idx = (finish_time < 0)

            jobs_slow_down[test_type].append(
                (finish_time[finished_idx] - enter_time[finished_idx]) / job_len[finished_idx]
            )
            work_complete[test_type].append(
                np.sum(job_len[finished_idx] * job_total_size[finished_idx])
            )
            work_remain[test_type].append(
                np.sum(job_len[unfinished_idx] * job_total_size[unfinished_idx])
            )
            job_len_remain[test_type].append(
                np.sum(job_len[unfinished_idx])
            )
            num_job_remain[test_type].append(
                len(job_len[unfinished_idx])
            )
            job_remain_delay[test_type].append(
                np.sum(pa.episode_max_length - enter_time[unfinished_idx])
            )

        env.seq_no = (env.seq_no + 1) % env.pa.num_ex

    # let's crunch some statistics!
    for test_type in test_types:
        disc_rew_ratios[test_type] = np.divide(all_discount_rews[test_type], all_discount_rews[ratio_comparers['optim_type']])
        knapsac_val_ratios[test_type] = np.divide(all_knapsac_vals[test_type], all_knapsac_vals[ratio_comparers['optim_type']])

    if pg_resume is not None:

        print "\nStatistics: \n"

        for test_type in test_types:
            if test_type == ratio_comparers["optim_type"]:
                continue

            max_disc_rew = np.max(disc_rew_ratios[test_type])
            min_disc_rew = np.min(disc_rew_ratios[test_type])
            avg_disc_rew = np.mean(disc_rew_ratios[test_type])
            avg_adj_disc_rew = np.mean(disc_rew_ratios[test_type]+1)
            var_disc_rew = np.var(disc_rew_ratios[test_type])
            var_adj_disc_rew = np.var(disc_rew_ratios[test_type]+1)

            max_knap_val = np.max(knapsac_val_ratios[test_type])
            min_knap_val = np.min(knapsac_val_ratios[test_type])
            avg_knap_val = np.mean(knapsac_val_ratios[test_type])
            avg_adj_knap_val = np.mean(knapsac_val_ratios[test_type]+1)
            var_knap_val = np.var(knapsac_val_ratios[test_type])
            var_adj_knap_val = np.var(knapsac_val_ratios[test_type]+1)

            print "Performance of " + test_type + " against " + ratio_comparers["optim_type"] + ":\n"

            print "Max Total Reward Ratio: \t\t" + str(max_disc_rew)
            print "Min Total Reward Ratio: \t\t" + str(min_disc_rew)
            print "Average of Total Reward Ratios: \t" + str(avg_disc_rew) + " (original) \t" + str(avg_adj_disc_rew) + " (adjusted)"
            print "Variance of Total Reward Ratios: \t" + str(var_disc_rew) + " (original) \t" + str(var_adj_disc_rew) + " (adjusted)\n"

            print "Max Knapsack Value Ratio: \t\t" + str(max_knap_val)
            print "Min Knapsack Value Ratio: \t\t" + str(min_knap_val)
            print "Average of Knapsack Value Ratios: \t" + str(avg_knap_val) + " (original) \t" + str(avg_adj_knap_val) + " (adjusted)"
            print "Variance of Knapsack Value Ratios: \t" + str(var_knap_val) + " (original) \t" + str(var_adj_knap_val) + " (adjusted)\n"

    # -- matplotlib colormap no overlap --
    if plot:
        num_colors = len(test_types)
        cm = plt.get_cmap('gist_rainbow')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_prop_cycle('color', [cm(1. * i / num_colors) for i in range(num_colors)])

        for test_type in test_types:
            val_perf_cdf = np.sort(knapsac_val_ratios[test_type])
            val_perf_yvals = np.arange(len(val_perf_cdf))/float(len(val_perf_cdf) - 1)
            ax.plot(val_perf_cdf, val_perf_yvals, linewidth=2, label=test_type)

        ax.legend(loc=4)
        ax.set_xlabel("Performance vs. KP", fontsize=20)
        ax.set_ylabel("CDF", fontsize=20)

        plt.savefig(pa.output_filename + "_perfratio_fig" + ".pdf")

    return all_discount_rews, all_knapsac_vals, jobs_slow_down


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
