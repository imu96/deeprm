import numpy as np
import parameters
from math import exp

max_res = 200
max_density = 20
min_density = 1

def psi(upper_bound, lower_bound, frac):
    threshold = ((upper_bound*exp(1)/lower_bound)**frac)*(lower_bound/exp(1))
    return threshold

def get_kp_action(machine, job_slot, upper_bound, lower_bound, cap=None):
    used_space = 0
    avbl_res = machine.avbl_slot[0, :]
    for i in xrange(len(job_slot.slot)):
        for res in xrange(machine.num_res):
            new_job = job_slot.slot[i]
            if new_job is None:
                continue

            density = float(new_job.len) / new_job.res_vec[i]
            used_space = machine.res_slot - avbl_res[res]
            capacity = machine.res_slot
            if cap is not None:
                used_space = cap - avbl_res[res]
                capacity = cap

            if new_job.res_vec[i] + used_space > capacity:
                continue;
            frac = used_space*1.0 / capacity
            psi_i = psi(upper_bound, lower_bound, frac);
            if density >=  psi_i:
                return res

    return machine.num_res

def get_packer_action(machine, job_slot):
        align_score = 0
        act = len(job_slot.slot)  # if no action available, hold

        for i in xrange(len(job_slot.slot)):
            new_job = job_slot.slot[i]
            if new_job is not None:  # there is a pending job

                avbl_res = machine.avbl_slot[:new_job.len, :]
                res_left = avbl_res - new_job.res_vec

                if np.all(res_left[:] >= 0):  # enough resource to allocate

                    tmp_align_score = avbl_res[0, :].dot(new_job.res_vec)

                    if tmp_align_score > align_score:
                        align_score = tmp_align_score
                        act = i
        return act


def get_sjf_action(machine, job_slot):
        sjf_score = 0
        act = len(job_slot.slot)  # if no action available, hold

        for i in xrange(len(job_slot.slot)):
            new_job = job_slot.slot[i]
            if new_job is not None:  # there is a pending job

                avbl_res = machine.avbl_slot[:new_job.len, :]
                res_left = avbl_res - new_job.res_vec

                if np.all(res_left[:] >= 0):  # enough resource to allocate

                    tmp_sjf_score = 1 / float(new_job.len)

                    if tmp_sjf_score > sjf_score:
                        sjf_score = tmp_sjf_score
                        act = i
        return act


def get_packer_sjf_action(machine, job_slot, knob):  # knob controls which to favor, 1 to packer, 0 to sjf

        combined_score = 0
        act = len(job_slot.slot)  # if no action available, hold

        for i in xrange(len(job_slot.slot)):
            new_job = job_slot.slot[i]
            if new_job is not None:  # there is a pending job

                avbl_res = machine.avbl_slot[:new_job.len, :]
                res_left = avbl_res - new_job.res_vec

                if np.all(res_left[:] >= 0):  # enough resource to allocate

                    tmp_align_score = avbl_res[0, :].dot(new_job.res_vec)
                    tmp_sjf_score = 1 / float(new_job.len)

                    tmp_combined_score = knob * tmp_align_score + (1 - knob) * tmp_sjf_score

                    if tmp_combined_score > combined_score:
                        combined_score = tmp_combined_score
                        act = i
        return act


def get_random_action(num_res):
    num_act = num_res + 1  # if no action available,
    act = np.random.randint(num_act)
    return act
