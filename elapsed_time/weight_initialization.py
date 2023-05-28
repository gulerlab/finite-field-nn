from mpi4py import MPI
import numpy as np
import random
from array import array
import math
import time
import sys
import gc
import os

import matplotlib as mpl
import matplotlib.pylab as plt
import pickle as pickle

from utils.mpc_function import *
from utils.polyapprox_function import *

#################################################
############ Distributed System Setting #########
#################################################

# for communication client distribution
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if len(sys.argv) == 1:
    if rank == 0:
        print("ERROR: please input the number of workers")
    exit()
else:
    N = int(sys.argv[1])

# for different cases of (K, T) pairs
N_case = 1
K_ = [int(N / 16)]
T_ = [int(N / 8)]
K = K_[0]
T = T_[0]

itemsize = MPI.DOUBLE.Get_size()
if rank == 0:
    nbytes = (N + 1) * itemsize
else:
    nbytes = 0
win = MPI.Win.Allocate_shared(nbytes, itemsize, comm=comm)

buf, itemsize = win.Shared_query(0)
assert itemsize == MPI.DOUBLE.Get_size()
buf = np.array(buf, dtype='B', copy=False)
ary = np.ndarray(buffer=buf, dtype='d', shape=(size,)) 

#################################################
################# Learning parameters ###########
#################################################
max_iter = 50
# set the seed of the random number generator for consistency
np.random.seed(42)
p = 2**26 - 5

# the bandwidth in bits
BW = 40_000_000  # 40Mbps

######################################################
################# Distributor -- rank-0 ##############
######################################################

if rank == 0:
    # print("### This is the Rank-0 Distributor ! ####")
    # print("00.Load in and Process the CIFAR-10 dataset.")

    # iterates over different settings of (K, T)
    for idx_case in range(N_case):

        # print("########### For the Case : K = ", K, " and T = ", T, " #########")
        # the size of dataset MINIST
        m = 50000
        d = 25088
        c = 10
        # print("01.Data Conversion : Real to Finite Field")
        for j in range(1, N + 1):
            # split the original dataset equally
            m_j = int(m / N)
            start_row = int(m_j * (j - 1))
            if j == N:
                m_j += (m % j)
            m_j = m_j - (m_j % K)
            # communicate the local dataset to the correspnding client
            comm.send(m_j, dest=j)  # send number of rows =  number of training samples
            comm.send(d, dest=j)  # send number of columns = number of features
            comm.send(c, dest=j)  # send the number of classes

        # print("02.Generation of all random matrices")
        comm.Barrier()

        # No Broadcasting step in Model Initialization !!


######################################################
################### Clients -- rank-j ################
######################################################

elif rank <= N:

    # print("### This is the client-", rank, " ####")

    for idx_case in range(N_case):

        ##################### Receive the F.F. dataset and the random mactrices #################
        # receive the raw dataset(local)
        m_i = comm.recv(source=0)  # number of rows =  number of training samples
        d = comm.recv(source=0)  # number of columns  = number of features
        c = comm.recv(source=0)  # total number of classes
        comm.Barrier()

        X_i = np.random.randint(p, size=(m_i, d))
        y_i = np.random.randint(p, size=(m_i, c))

        # receive the random matrices for encoding
        d_hidden = 128
        d_hidden = d_hidden + (N - T) - np.mod(d_hidden, (N - T))
        V_i_hidden = np.random.randint(p, size=(T, int(d_hidden / (N - T)), 1))
        d_out = 10
        if (N - T) < 10:
            d_out = d_out + (N - T) - np.mod(d_out, (N - T))
        else:
            d_out = (N - T)
        V_i_out = np.random.randint(p, size=(T, int(d_out / (N - T)), 1))

        delta_time = 0.0

        ############################################
        #         Computation of M_matrix          #
        ############################################
        t_compute_M_matrix = time.time()
        M_matrix = np.empty((N, N - T), dtype="int64")
        for t in range(N):
            if t == 0:
                M_matrix[t, :] = np.ones(N - T, dtype="int64")
            else:
                # modular operation for Overflow problem
                M_matrix[t, :] = np.mod(np.array([i ** t for i in range(1, N - T + 1)]), p)
        M_matrix = np.reshape(M_matrix, (N - T, N))
        t_compute_M_matrix = time.time() - t_compute_M_matrix

        ############################################
        #       Stage-3 : Model Initialization     #
        ############################################
        # the initial model of the hidden layer -- model paramters
        t_compute_w_hidden = time.time()
        w_hidden_i = (1 / float(60_000)) * np.random.rand(int(d_hidden / (N - T)), 1)
        w_hidden_i = my_q(w_hidden_i, 0, p)
        w_hidden_i = np.transpose(np.tile(np.transpose(w_hidden_i), K))
        w_hidden_i = LCC_encoding_w_Random(w_hidden_i, V_i_hidden, N, K, T, p)
        t_compute_w_hidden = time.time() - t_compute_w_hidden
        # the initial model of the hidden layer -- Bias term
        t_compute_b_hidden = time.time()
        b_hidden_i = (1 / float(60_000)) * np.random.rand(int(d_hidden / (N - T)), 1)
        b_hidden_i = my_q(b_hidden_i, 0, p)
        b_hidden_i = np.transpose(np.tile(np.transpose(b_hidden_i), K))
        b_hidden_i = LCC_encoding_w_Random(b_hidden_i, V_i_hidden, N, K, T, p)
        t_compute_b_hidden = time.time() - t_compute_b_hidden
        # the corresponding communi volume -- model
        w_hidden_SS_T = np.empty((N, int(d_hidden / (N - T))), dtype='int64')
        volume_w_hidden = time.time()
        delta_time = 0.0
        for j in range(1, N + 1):
            if rank == j:
                w_hidden_SS_T[j - 1] = np.reshape(w_hidden_i[j - 1, :, :], int(d_hidden / (N - T)))
                for j_others in range(1, N + 1):
                    if j_others == rank: continue
                    # send the SS of the local initial model
                    delta_time += int(d_hidden / (N - T)) * 64 / BW
                    comm.Send(np.reshape(w_hidden_i[j_others - 1, :, :], int(d_hidden / (N - T))), dest=j_others)
            else:
                data = np.empty(int(d_hidden / (N - T)), dtype='int64')
                comm.Recv(data, source=j)
                if rank == N:
                    delta_time += (N - 1) * int(d_hidden / (N - T)) * 64 / BW
                else:
                    if j == N:
                        delta_time += rank * int(d_hidden / (N - T)) * 64 / BW
                    else:
                        delta_time += (N - 1) * int(d_hidden / (N - T)) * 64 / BW
                w_hidden_SS_T[j - 1] = data
        volume_w_hidden = time.time() - volume_w_hidden + delta_time
        # the corresponding communi volume -- model
        b_hidden_SS_T = np.empty((N, int(d_hidden / (N - T))), dtype='int64')
        volume_b_hidden = time.time()
        delta_time = 0.0
        for j in range(1, N + 1):
            if rank == j:
                b_hidden_SS_T[j - 1] = np.reshape(b_hidden_i[j - 1, :, :], int(d_hidden / (N - T)))
                for j_others in range(1, N + 1):
                    if j_others == rank: continue
                    # send the SS of the local initial model
                    delta_time += int(d_hidden / (N - T)) * 64 / BW
                    comm.Send(np.reshape(b_hidden_i[j_others - 1, :, :], int(d_hidden / (N - T))), dest=j_others)
            else:
                data = np.empty(int(d_hidden / (N - T)), dtype='int64')
                comm.Recv(data, source=j)
                if rank == N:
                    delta_time += (N - 1) * int(d_hidden / (N - T)) * 64 / BW
                else:
                    if j == N:
                        delta_time += rank * int(d_hidden / (N - T)) * 64 / BW
                    else:
                        delta_time += (N - 1) * int(d_hidden / (N - T)) * 64 / BW
                b_hidden_SS_T[j - 1] = data
        volume_b_hidden = time.time() - volume_b_hidden + delta_time
        # compute the SS of the whole initialized model
        t_w_init = time.time()
        w_hidden_SS_T = np.dot(M_matrix, w_hidden_SS_T)
        w_hidden_SS_T = np.reshape(w_hidden_SS_T, ((N - T) * int(d_hidden / (N - T)), 1))
        t_compute_w_hidden += time.time() - t_w_init
        t_b_init = time.time()
        b_hidden_SS_T = np.dot(M_matrix, b_hidden_SS_T)
        b_hidden_SS_T = np.reshape(b_hidden_SS_T, ((N - T) * int(d_hidden / (N - T)), 1))
        t_compute_b_hidden += time.time() - t_b_init

        # the initial model of the hidden layer -- model paramters
        t_compute_w_out = time.time()
        w_out_i = (1 / float(60_000)) * np.random.rand(int(d_out / (N - T)), 1)
        w_out_i = my_q(w_out_i, 0, p)
        w_out_i = np.transpose(np.tile(np.transpose(w_out_i), K))
        w_out_i = LCC_encoding_w_Random(w_out_i, V_i_out, N, K, T, p)
        t_compute_w_out = time.time() - t_compute_w_out
        # the initial model of the hidden layer -- Bias term
        t_compute_b_out = time.time()
        b_out_i = (1 / float(60_000)) * np.random.rand(int(d_out / (N - T)), 1)
        b_out_i = my_q(b_out_i, 0, p)
        b_out_i = np.transpose(np.tile(np.transpose(b_out_i), K))
        b_out_i = LCC_encoding_w_Random(b_out_i, V_i_out, N, K, T, p)
        t_compute_b_out = time.time() - t_compute_b_out
        # the corresponding communi volume -- model
        w_out_SS_T = np.empty((N, int(d_out / (N - T))), dtype='int64')
        volume_w_out = time.time()
        delta_time = 0.0
        for j in range(1, N + 1):
            if rank == j:
                w_out_SS_T[j - 1] = np.reshape(w_out_i[j - 1, :, :], int(d_out / (N - T)))
                for j_others in range(1, N + 1):
                    if j_others == rank: continue
                    # send the SS of the local initial model
                    delta_time += int(d_out / (N - T)) * 64 / BW
                    comm.Send(np.reshape(w_out_i[j_others - 1, :, :], int(d_out / (N - T))), dest=j_others)
            else:
                data = np.empty(int(d_out / (N - T)), dtype='int64')
                comm.Recv(data, source=j)
                if rank == N:
                    delta_time += (N - 1) * int(d_out / (N - T)) * 64 / BW
                else:
                    if j == N:
                        delta_time += rank * int(d_out / (N - T)) * 64 / BW
                    else:
                        delta_time += (N - 1) * int(d_out / (N - T)) * 64 / BW
                w_out_SS_T[j - 1] = data
        volume_w_out = time.time() - volume_w_out + delta_time
        # the corresponding communi volume -- model
        b_out_SS_T = np.empty((N, int(d_out / (N - T))), dtype='int64')
        volume_b_out = time.time()
        delta_time = 0.0
        for j in range(1, N + 1):
            if rank == j:
                b_out_SS_T[j - 1] = np.reshape(b_out_i[j - 1, :, :], int(d_out / (N - T)))
                for j_others in range(1, N + 1):
                    if j_others == rank: continue
                    # send the SS of the local initial model
                    delta_time += int(d_out / (N - T)) * 64 / BW
                    comm.Send(np.reshape(b_out_i[j_others - 1, :, :], int(d_out / (N - T))), dest=j_others)
            else:
                data = np.empty(int(d_out / (N - T)), dtype='int64')
                # print(data.shape)
                comm.Recv(data, source=j)
                if rank == N:
                    delta_time += (N - 1) * int(d_out / (N - T)) * 64 / BW
                else:
                    if j == N:
                        delta_time += rank * int(d_out / (N - T)) * 64 / BW
                    else:
                        delta_time += (N - 1) * int(d_out / (N - T)) * 64 / BW
                # b_hidden_SS_T[j-1] = data
        volume_b_out = time.time() - volume_b_out + delta_time
        # compute the SS of the whole initialized model
        t_w_init = time.time()
        w_out_SS_T = np.dot(M_matrix, w_out_SS_T)
        w_out_SS_T = np.reshape(w_out_SS_T, ((N - T) * int(d_out / (N - T)), 1))
        t_compute_w_out += time.time() - t_w_init
        t_b_init = time.time()
        b_out_SS_T = np.dot(M_matrix, b_out_SS_T)
        b_out_SS_T = np.reshape(b_out_SS_T, ((N - T) * int(d_out / (N - T)), 1))
        t_compute_b_out += time.time() - t_b_init

        # accumulate for all the model it is running
        # print("############ For the client-", rank, " ##########")
        # print("# W hidden : ", t_compute_w_hidden * d)
        # print("# b hidden : ", t_compute_b_hidden)
        # print("# commu W hidden : ", volume_w_hidden * d)
        # print("# commu b hidden : ", volume_b_hidden)
        # print("# W out : ", t_compute_w_out * d_hidden)
        # print("# b out : ", t_compute_b_out)
        # print("# commu W out : ", volume_w_out * d_hidden)
        # print("# commu b out : ", volume_b_out)
        # print("#################################################")

        # save the records to a separate file
        time_records_compute = np.array([t_compute_w_hidden*d, t_compute_b_hidden, t_compute_w_out*d_hidden, t_compute_b_out])
        time_records_comm = np.array([volume_w_hidden*d, volume_b_hidden, volume_w_out*d_hidden, volume_b_out])
        total_time_elapsed = np.sum(time_records_compute) + np.sum(time_records_comm)
        ary[rank] = total_time_elapsed
        # print('{}, {}'.format(rank, total_time_elapsed))
        # np.savetxt("testing_results/" + str(N) + "_case/PICO_NN_K-" + str(K) + "_T-" + str(T) + "_client-" + str(
        #     rank) + "_MINIST_40BW_ModelInit_compute.txt", time_records_compute)
        # np.savetxt("testing_results/" + str(N) + "_case/PICO_NN_K-" + str(K) + "_T-" + str(T) + "_client-" + str(
        #     rank) + "_MINIST_40BW_ModelInit_comm.txt", time_records_comm)

comm.Barrier()

if rank == 0:
    # print(ary)
    with open('weight_initialization_cifar10_vgg.txt'.format(N), 'a') as fp:
        fp.write('{}, {}\n'.format(N, np.amax(ary[1:])))
    print(np.amax(ary))


