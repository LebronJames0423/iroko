from multiprocessing import Array
from ctypes import c_ulong, c_ubyte
import numpy as np
import copy
import os

from dc_gym.monitor.iroko_monitor import BandwidthCollector
from dc_gym.monitor.iroko_monitor import QueueCollector
from dc_gym.monitor.iroko_monitor import FlowCollector
from dc_gym.monitor.iroko_monitor import RTTCollector
from dc_gym.monitor.iroko_monitor import RCV_RTTCollector
from dc_gym.monitor.iroko_monitor import CWNDCollector
from iroko_reward import RewardFunction


def shmem_to_nparray(shmem_array, dtype):
    return np.frombuffer(shmem_array.get_obj(), dtype=dtype)


class StateManager:
    STATS_DICT = {"backlog": 0, "olimit": 1,
                  "drops": 2, "bw_rx": 3, "bw_tx": 4}
    STATS_DICT_2 = {"rtt": 0, "rcv_rtt": 1, "cwnd": 2, "drops": 3}
    REWARD_MODEL = ["backlog", "action"]
    STATS_KEYS = ["backlog"]
    STATS_KEYS_2 = ["rtt", "rcv_rtt", "cwnd", "drops"]
    DELTA_KEYS = []
    COLLECT_FLOWS = False
    __slots__ = ["num_features", "num_ports", "deltas", "prev_stats",
                 "stats_file", "data", "dopamin", "stats", "flow_stats",
                 "procs"]

    def __init__(self, topo_conf, config):
        sw_ports = topo_conf.get_sw_ports()
        self.num_ports = len(sw_ports)
        self.deltas = None
        self.prev_stats = None
        self._set_feature_length(len(topo_conf.host_ips))
        self._init_stats_matrices(self.num_ports, len(topo_conf.host_ips))
        self._spawn_collectors(sw_ports, topo_conf.host_ips)
        self.dopamin = RewardFunction(topo_conf.host_ctrl_map,
                                      sw_ports, self.REWARD_MODEL,
                                      topo_conf.MAX_QUEUE,
                                      topo_conf.MAX_CAPACITY, self.STATS_DICT)
        self._set_data_checkpoints(config)
        self.topo1 = [-1.402699, 0.344527, 0.938987, \
                     -1.398684, 0.338663, 0.935212, \
                     -0.744777, 0.649759, 1.074763, \
                     -0.745094, 0.653848, 1.070479, \
                     0.694525, -0.970129, 1.288484, \
                     3.588135, -2.813144, 5.681612, \
                     0.637676, -0.091677, 1.093513, \
                     2.959620, -1.145467, 4.797094, \
                     -0.698438, -6.985178, 3.333977, \
                     2.022099, -5.508847, 5.079647, \
                     -1.801459, -5.486279, 1.914149, \
                     -0.572947, -1.478012, 0.495691, \
                     0.182531, 2.780981, 5.558248, \
                     0.187680, 2.780330, 5.571474, \
                     -5.439290, -3.345211, 1.658321, \
                     -5.422459, -3.340781, 1.652555, \
                     -5.179317, 3.486463, 5.877534, \
                     -5.144007, 3.467831, 5.829274, \
                     -6.996850, 1.514256, 4.843931, \
                     -7.017985, 1.555845, 4.890795]

        self.topo2 = [-1.169679, 0.747148, 0.952519, \
                     -1.164962, 0.747387, 0.958316, \
                     -0.773638, 0.950160, 1.252210, \
                     -0.773540, 0.953502, 1.248539, \
                     -1.016529, -4.715937, 2.693734, \
                     -1.000568, -4.598229, 2.613090, \
                     1.127291, -0.352509, 1.829293, \
                     2.765765, -1.462628, 4.648371, \
                     0.903844, -1.881926, 2.257098, \
                     3.132540, -3.363417, 5.957943, \
                     -1.811017, -3.581302, 1.722362, \
                     -1.232868, -1.863409, 0.884350, \
                     1.536948, 2.662853, 5.791492, \
                     1.548453, 2.673847, 5.824697, \
                     -5.983416, -1.410670, 2.385093, \
                     -5.966656, -1.410419, 2.378438, \
                     -4.046757, 4.819962, 6.078331, \
                     -4.003929, 4.784531, 6.023712, \
                     -6.094141, 3.688973, 5.170659, \
                     -6.115153, 3.707456, 5.200839]

        self.topo3 = [1, 2, 3, \
                      4, 5, 6, \
                      7, 8, 9, \
                      2, 3, 1, \
                      5, 6, 4, \
                      8, 9, 7, \
                      3, 1, 2, \
                      6, 4, 5, \
                      9, 7, 8]
    def terminate(self):
        self.flush()
        self._terminate_collectors()
        self.stats_file.close()

    def reset(self):
        pass        # self.flush()

    def _set_feature_length(self, num_hosts):
        self.num_features = len(self.STATS_KEYS)
        self.num_features += len(self.DELTA_KEYS)
        if self.COLLECT_FLOWS:
            # There are two directions for flows, src and destination
            self.num_features += num_hosts * 2

    def get_feature_length(self):
        return self.num_features

    def _init_stats_matrices(self, num_ports, num_hosts):
        self.stats = None
        self.stats_2 = None
        self.flow_stats = None
        self.procs = []
        # Set up the shared stats matrix
        stats_arr_len = num_ports * len(self.STATS_DICT)
        stats_2_arr_len = len(self.STATS_DICT_2)
        mp_stats = Array(c_ulong, stats_arr_len)
        mp_stats_2 = Array(c_ulong, stats_2_arr_len)
        np_stats = shmem_to_nparray(mp_stats, np.int64)
        np_stats_2 = shmem_to_nparray(mp_stats_2, np.int64)
        self.stats = np_stats.reshape((num_ports, len(self.STATS_DICT)))
        self.stats_2 = np_stats_2.reshape((1, len(self.STATS_DICT_2)))
        # Set up the shared flow matrix
        flow_arr_len = num_ports * num_hosts * 2
        mp_flows = Array(c_ubyte, flow_arr_len)
        np_flows = shmem_to_nparray(mp_flows, np.uint8)
        self.flow_stats = np_flows.reshape((num_ports, 2, num_hosts))
        # Save the initialized stats matrix to compute deltas
        self.prev_stats = self.stats.copy()
        self.deltas = np.zeros(shape=(num_ports, len(self.STATS_DICT)))

    def _spawn_collectors(self, sw_ports, host_ips):
        # Launch an asynchronous queue collector

#        os.system("sudo mn -c")


        proc = QueueCollector(sw_ports, self.stats, self.STATS_DICT, self.stats_2, self.STATS_DICT_2)
        proc.start()
        self.procs.append(proc)
#        print('------------------++++++++++++',self.stats)
        # Launch an asynchronous bandwidth collector
        proc = BandwidthCollector(sw_ports, self.stats, self.STATS_DICT)
        proc.start()
        self.procs.append(proc)
#        print('-----------------++++++++++++', self.stats)


        # Launch an asynchronous rtt collector
        proc = RTTCollector(sw_ports, self.stats_2, self.STATS_DICT_2)
        proc.start()
        self.procs.append(proc)
#        print('-----------------++++++++++++', self.stats)


        # Launch an asynchronous rcv_rtt collector
        proc = RCV_RTTCollector(sw_ports, self.stats_2, self.STATS_DICT_2)
        proc.start()
        self.procs.append(proc)
#        print('-----------------++++++++++++', self.stats)


        # Launch an asynchronous cwnd collector
        proc = CWNDCollector(sw_ports, self.stats_2, self.STATS_DICT_2)
        proc.start()
        self.procs.append(proc)
#        print('-----------------++++++++++++', self.stats)


        # Launch an asynchronous flow collector
        proc = FlowCollector(sw_ports, host_ips, self.flow_stats)
        proc.start()
        self.procs.append(proc)
#        print('------------------+++++++++++', self.stats)

    def _set_data_checkpoints(self, conf):
        self.data = {}
        data_dir = conf["output_dir"]
        agent = conf["agent"]

        # define file name
        runtime_name = "%s/runtime_statistics_%s.npy" % (data_dir, agent)
        self.stats_file = open(runtime_name, 'wb+')
        self.data["reward"] = []
        self.data["action_reward"] = []
        self.data["bw_reward"] = []
        self.data["queue_reward"] = []
        self.data["std_dev_reward"] = []
        self.data["actions"] = []
        self.data["stats"] = []
        self.data["stats_2"] = []

    def _terminate_collectors(self):
        for proc in self.procs:
            if proc is not None:
                proc.terminate()

    def _compute_deltas(self, num_ports, stats_prev, stats_now):
        for iface_index in range(num_ports):
            for delta_index, stat in enumerate(self.STATS_DICT.keys()):
                stat_index = self.STATS_DICT[stat]
                prev = stats_prev[iface_index][stat_index]
                now = stats_now[iface_index][stat_index]
                self.deltas[iface_index][delta_index] = now - prev

    def observe(self):
        obs = []
        obs_2 = []
        # retrieve the current deltas before updating total values
        self._compute_deltas(self.num_ports, self.prev_stats, self.stats)
        self.prev_stats = self.stats.copy()
        # Create the data matrix for the agent based on the collected stats
        for index in range(self.num_ports):
            state = []
            for key in self.STATS_KEYS:
                state.append(int(self.stats[index][self.STATS_DICT[key]]))
            for key in self.DELTA_KEYS:
                state.append(int(self.deltas[index][self.STATS_DICT[key]]))
            if self.COLLECT_FLOWS:
                state.extend(self.flow_stats[index])
            # print("State %d: %s " % (index, state))
            obs.append(np.array(state))
         
        state = []
        for key in self.STATS_KEYS_2:
            state.append(int(self.stats_2[0][self.STATS_DICT_2[key]]))
        obs_2.append(np.array(state))        
#        for num in self.topo2:
#            temp = []
#            temp.append(num)
#            obs.append(np.array(temp))
#        print("obs-----------------", obs)
        # Save collected data
        arr_temp = copy.deepcopy(self.stats)
        arr_temp_2 = copy.deepcopy(self.stats_2)
#        arr_temp = np.append(arr_temp, self.topo2)
        self.data["stats"].append(arr_temp)
        self.data["stats_2"].append(arr_temp_2)
#        print("-------------------", self.data["stats"][-1])
#        print("+++++++++++++++++++", self.data["stats_2"][-1])
        return np.array(obs), np.array(obs_2)

    def compute_reward(self, curr_action):
        # Compute the reward
        reward, action_reward, bw_reward, queue_reward, std_dev_reward = self.dopamin.get_reward(self.stats, self.deltas, curr_action)
        self.data["reward"].append(reward)
        self.data["action_reward"].append(action_reward)
        self.data["bw_reward"].append(bw_reward)
        self.data["queue_reward"].append(queue_reward)
        self.data["std_dev_reward"].append(std_dev_reward)
        self.data["actions"].append(curr_action)
        return reward

    def flush(self):
        print("Saving statistics...")
        np.save(self.stats_file, np.array(self.data))
        self.stats_file.flush()
        for key in self.data.keys():
            del self.data[key][:]

