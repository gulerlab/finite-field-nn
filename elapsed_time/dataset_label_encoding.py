import sys
import os
# import time

from utils.mpc_function import *
from utils.polyapprox_function import *

#################################################
############ Distributed System Setting #########
#################################################

# for communication client distribution
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print(rank, size)

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
    print("### This is the Rank-0 Distributor ! ####")
    print("00.Load in and Process the CIFAR-10 dataset.")

    # iterates over different settings of (K, T)
    for idx_case in range(N_case):

        # the particular case of (K, T) pair
        K = K_[idx_case]
        T = T_[idx_case]
        print("########### For the Case : K = ", K, " and T = ", T, " #########")
        # the size of dataset MNIST/Fashion-MNIST
        m = 60000
        d = 784
        c = 10
        print("01.Data Conversion : Real to Finite Field")
        for j in range(1, N + 1):
            # split the original dataset equally
            m_j = int(m / N)
            start_row = int(m_j * (j - 1))
            if j == N:
                m_j += (m % j)
            m_j = m_j - (m_j % K)
            # communicate the local dataset to the corresponding client
            comm.send(m_j, dest=j)  # send number of rows =  number of training samples
            comm.send(d, dest=j)  # send number of columns = number of features
            comm.send(c, dest=j)  # send the number of classes

        print("02.Generation of all random matrices")
        comm.Barrier()


######################################################
################### Clients -- rank-j ################
######################################################

elif rank <= N:

    print("### This is the client-", rank, " ####")

    for idx_case in range(N_case):

        ##################### Receive the F.F. dataset and the random matrices #################
        # receive the raw dataset(local)
        m_i = comm.recv(source=0)  # number of rows =  number of training samples
        d = comm.recv(source=0)  # number of columns  = number of features
        c = comm.recv(source=0)  # total number of classes
        comm.Barrier()

        X_i = np.random.randint(p, size=(m_i, d))
        y_i = np.random.randint(p, size=(m_i, c))

        # receive the random matrices for encoding -- data features
        V_i = np.random.randint(p, size=(T, int(m_i / K), d))
        # receive the random matrices for encoding -- labels
        a_i = np.random.randint(p, size=(T, int(m_i / K), c))

        r1_trunc_i = np.random.randint(p, size=(d, 1))
        r2_trunc_i = np.random.randint(p, size=(d, 1))

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
        #       Stage-1 : Dataset Encoding         #
        ############################################
        # computation of LCC of the X_i
        t_compute_X_LCC = time.time()
        X_SS_T_i = LCC_encoding_w_Random(X_i, V_i, N, K, T, p)
        t_compute_X_LCC = time.time() - t_compute_X_LCC
        # computation of LCC of the y_i
        t_compute_Y_LCC = time.time()
        y_SS_T_i = LCC_encoding_w_Random(y_i, a_i, N, K, T, p)
        t_compute_Y_LCC = time.time() - t_compute_Y_LCC
        # communicate the SS of X_i for [X]_i, and the SS of y_scale_i for [y]_i
        X_SS_T = []
        y_SS_T = []
        volume_X_LCC = time.time()
        volume_Y_LCC = time.time()
        delta_time_X_LCC = 0.0
        delta_time_Y_LCC = 0.0
        for j in range(1, N + 1):
            if rank == j:
                X_SS_T.append(X_SS_T_i[j - 1, :, :])
                y_SS_T.append(y_SS_T_i[j - 1, :, :])
                for j_others in range(1, N + 1):
                    if j_others == rank: continue
                    # send the size of the local dataset
                    comm.send(m_i, dest=j_others)
                    # send the SS of the local dataset -- images
                    delta_time_X_LCC += (m_i / K * d * 64 / BW)
                    comm.Send(np.reshape(X_SS_T_i[j_others - 1, :, :], int(m_i / K) * d), dest=j_others)
                    # send the SS of the local dataset -- labels
                    delta_time_Y_LCC += (m_i / K * c * 64 / BW)
                    comm.Send(np.reshape(y_SS_T_i[j_others - 1, :, :], int(m_i / K) * c), dest=j_others)
                X_SS_T_i = None
                y_SS_T_i = None
            else:
                m_j = comm.recv(source=j)
                # receive the SS of data points
                data_SS = np.empty(int(m_j / K) * d, dtype='int64')
                comm.Recv(data_SS, source=j)
                if rank == N:
                    delta_time_X_LCC += (N - 1) * int(m_j / K * d) * 64 / BW
                else:
                    if j == N:
                        delta_time_X_LCC += rank * int(m_j / K * d) * 64 / BW
                    else:
                        delta_time_X_LCC += (N - 1) * int(m_j / K * d) * 64 / BW
                # receive the SS of labels
                label_SS = np.empty(int(m_j / K) * c, dtype='int64')
                comm.Recv(label_SS, source=j)
                if rank == N:
                    delta_time_Y_LCC += (N - 1) * int(m_j / K * c) * 64 / BW
                else:
                    if j == N:
                        delta_time_Y_LCC += rank * int(m_j / K * c) * 64 / BW
                    else:
                        delta_time_Y_LCC += (N - 1) * int(m_j / K * c) * 64 / BW
                # reshape to the normal shape
                data_SS = np.reshape(data_SS, (int(m_j / K), d)).astype('int64')
                label_SS = np.reshape(label_SS, (int(m_j / K), c)).astype('int64')
                X_SS_T.append(data_SS)
                y_SS_T.append(label_SS)
        # record the communication time-cost
        volume_X_LCC = time.time() - volume_X_LCC + delta_time_X_LCC
        volume_Y_LCC = time.time() - volume_Y_LCC + delta_time_Y_LCC
        # concatination for overall X_LCC
        delta_time = time.time()
        X_SS_T = np.concatenate(np.array(X_SS_T))
        t_compute_X_LCC += (time.time() - delta_time)
        # time-cost for concatination for overall Y_LCC
        delta_time = time.time()
        y_SS_T = np.concatenate(np.array(y_SS_T))
        t_compute_Y_LCC += (time.time() - delta_time)

        # save the records to a separate file
        time_records_compute = np.array([t_compute_M_matrix, t_compute_X_LCC, t_compute_Y_LCC])
        time_records_comm = np.array([volume_X_LCC, volume_Y_LCC])
        np.savetxt("testing_results/" + str(N) + "_case/PICO_NN_K-" + str(K) + "_T-" + str(T) + "_client-" + str(
            rank) + "_MINIST_40BW_DatasetEncoding_compute.txt", time_records_compute)
        np.savetxt("testing_results/" + str(N) + "_case/PICO_NN_K-" + str(K) + "_T-" + str(T) + "_client-" + str(
            rank) + "_MINIST_40BW_DatasetEncoding_comm.txt", time_records_comm)




