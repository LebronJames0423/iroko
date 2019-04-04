from __future__ import print_function
import os
import subprocess
import datetime
import time
import json
import socket
from plot import plot


# set up paths
exec_dir = os.getcwd()
file_dir = os.path.dirname(__file__)
INPUT_DIR = file_dir + '/inputs'
OUTPUT_DIR = exec_dir + '/results'
PLOT_DIR = exec_dir + '/plots'
#RL_ALGOS = ["PPO", "PG", "DDPG"]
#TCP_ALGOS = ["DCTCP", "TCP_NV"]
RL_ALGOS = ["DDPG"]
TCP_ALGOS = []
ALGOS = TCP_ALGOS + RL_ALGOS
#TRANSPORT = ["udp", "tcp"]
TRANSPORT = ["udp"]
RUNS = 1
STEPS = 28000
#TOPO = "dumbbell"
TOPO = "fattree"
TUNE = False
RESTORE = False
RESTORE_PATH = exec_dir + "/checkpoint-1"


def check_dir(directory):
    # create the folder if it does not exit
    if not directory == '' and not os.path.exists(directory):
        print("Folder %s does not exist! Creating..." % directory)
        os.makedirs(directory)


def generate_testname(output_dir):
    n_folders = 0
    if os.path.isdir(output_dir):
        f_list = os.listdir(output_dir)
        n_folders = len(f_list)
    # Host name and a time stamp
    testname = "%s_%s" % (socket.gethostname(), n_folders)
    return testname


def dump_config(path):
