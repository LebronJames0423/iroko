from __future__ import print_function
from multiprocessing import Array
import subprocess
from ctypes import c_ulong, c_ubyte
import numpy as np
import time
from dc_gym.env_base import BaseEnv
from dc_gym.monitor.iroko_monitor import RTTCollector
from dc_gym.control.iroko_bw_control import BandwidthController
import os


def shmem_to_nparray(shmem_array, dtype):
    return np.frombuffer(shmem_array.get_obj(), dtype=dtype)

class DCEnv(BaseEnv):
    WAIT = 0.0      # amount of seconds the agent waits per iteration
    __slots__ = ["bw_ctrl"]

    def __init__(self, conf):
        BaseEnv.__init__(self, conf)
        self.bw_ctrl = BandwidthController(self.topo.host_ctrl_map)
#        print(self.topo.host_ctrl_map)
    def step(self, action):
        STATS_DICT_2 = {"rtt": 0, "rcv_rtt": 1, "cwnd": 2, "drops": 3}
        BaseEnv.step(self, action)
        # if the traffic generator still going then the simulation is not over
        # let the agent predict bandwidth based on all previous information
        # perform actions
        done = not self.is_traffic_proc_alive()
        pred_bw = action * self.topo.MAX_CAPACITY
#        print("----------------------", action*100)
        
        # action 1
        self.bw_ctrl.broadcast_bw(pred_bw, self.topo.host_ctrl_map)
        # action 2
#        cmd0 = ("sudo tc qdisc add dev sw1-eth2 root netem delay %dms" % (action[0]*100))
#        cmd1 = ("sudo tc qdisc add dev sw1-eth3 root netem delay %dms" % (action[1]*100))
#        cmd2 = ("sudo tc qdisc add dev sw2-eth2 root netem delay %dms" % (action[2]*100))
#        cmd3 = ("sudo tc qdisc add dev sw2-eth3 root netem delay %dms" % (action[3]*100))
#        os.system(cmd0)
#        os.system(cmd1)
#        os.system(cmd2)
#        os.system(cmd3)

        # observe for WAIT seconds minus time needed for computation
        max_sleep = max(self.WAIT - (time.time() - self.start_time), 0)
        time.sleep(max_sleep)
        self.start_time = time.time()

        obs, obs_2 = self.state_man.observe()
#        print('-------------obs', obs, obs_2)
        self.reward = self.state_man.compute_reward(pred_bw)

        cmd = "ss -ti | grep -Eo ' rtt:[0-9]*\.[0-9]*' | grep -Eo '[0-9]*\.[0-9]*'"
        proc = subprocess.Popen([cmd], stdout=subprocess.PIPE, shell=True)
        proc.wait()
        output, _ = proc.communicate()
        output = output.decode()

        rtt = output.split('\n')
        sum = 0
        for i in range(len(rtt)-1):
            sum = sum + float(rtt[i])*1000
        avg_rtt = sum/(len(rtt)-1)
        self.reward_2 = -1 * avg_rtt
#        print("----------------------", avg_rtt)

        return obs.flatten(), self.reward, done, {}
