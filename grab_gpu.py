import logging
import os
import signal
import sys
import torch
import torch.nn as nn
import time
import numpy as np

from subprocess import Popen, PIPE

LOG_FORMAT = f'(PID {os.getpid()}) %(asctime)s %(levelname)s >>> %(message)s'
logging.basicConfig(level=logging.INFO,
                    format=LOG_FORMAT,
                    handlers=[logging.StreamHandler(sys.stdout)])

class GPU:
    def __init__(self, ID, uuid, load, memoryTotal, memoryUsed,
                 memoryFree, driver, gpu_name, serial, display_mode,
                 display_active, temp_gpu):
        self.id = ID
        self.uuid = uuid
        self.load = load
        self.memoryUtil = float(memoryUsed) / float(memoryTotal)
        self.memoryTotal = memoryTotal
        self.memoryUsed = memoryUsed
        self.memoryFree = memoryFree
        self.driver = driver
        self.name = gpu_name
        self.serial = serial
        self.display_mode = display_mode
        self.display_active = display_active
        self.temperature = temp_gpu

def safeFloatCast(strNumber):
    try:
        number = float(strNumber)
    except ValueError:
        number = float('nan')
    return number

def getGPUs():
    # Get ID, processing and memory utilization for all GPUs
    try:
        p = Popen(["nvidia-smi",
                   "--query-gpu=index,uuid,utilization.gpu,memory.total,"
                   "memory.used,memory.free,driver_version,name,gpu_serial,"
                   "display_active,display_mode,temperature.gpu",
                   "--format=csv,noheader,nounits"],
                  stdout=PIPE)
        stdout, stderror = p.communicate()
    except Exception as err:
        print("Error: failed to get GPU infos\n", err)
        return []

    output = stdout.decode('UTF-8').strip()
    # Split by line break
    lines = output.split(os.linesep)
    numDevices = len(lines)
    GPUs = []
    for g in range(numDevices):
        line = lines[g]
        vals = line.split(', ')
        deviceIds = int(vals[0])
        uuid = vals[1]
        gpuUtil = safeFloatCast(vals[2])/100
        memTotal = safeFloatCast(vals[3])
        memUsed = safeFloatCast(vals[4])
        memFree = safeFloatCast(vals[5])
        driver = vals[6]
        gpu_name = vals[7]
        serial = vals[8]
        display_active = vals[9]
        display_mode = vals[10]
        temp_gpu = safeFloatCast(vals[11])
        gpu = GPU(deviceIds, uuid, gpuUtil, memTotal, memUsed, memFree, driver,
                  gpu_name, serial, display_mode, display_active, temp_gpu)
        GPUs.append(gpu)
    return GPUs  # (deviceIds, gpuUtil, memUtil)

class NaiveNetwork(nn.Module):
    def __init__(self, gpu_id):
        super(NaiveNetwork, self).__init__()
        self.conv = nn.Conv2d(32, 32, 3, 1, 1)
        self.data = torch.ones(256, 32, 128, 128).to(gpu_id)
        self.to(gpu_id)

    def forward(self):
        for i in range(10000):
            self.conv(self.data)

def get_gpu_load(gpu_id_num):
    gpus = getGPUs()
    gpu = gpus[gpu_id_num]
    return gpu.load

def get_gpu_memoryUsed(gpu_id_num):
    gpus = getGPUs()
    gpu = gpus[gpu_id_num]
    return gpu.memoryUsed

def occupy_gpu(model, thres, gpu_id_num):
    max_gpu_memUsed = 0
    while True:
        model.forward()
        a = time.time()
        while True:
            b = time.time()
            gpu_memUsed = get_gpu_memoryUsed(gpu_id_num)
            max_gpu_memUsed = np.max((max_gpu_memUsed, gpu_memUsed))
            if (b - a) > 0.5:
                break
        if max_gpu_memUsed > thres:
            logging.info("Detect task on GPU[{}]. Break.".format(gpu_id_num))
            break

def subprocess(gpu_id_num):
    thres = 3000
    model = NaiveNetwork(gpu_id_num)

    monitor = False
    while True:
        time.sleep(1)
        start = time.time()
        max_gpu_memUsed = 0
        while True:
            time.sleep(0.1)   # Calm down GPU
            gpu_memUsed = get_gpu_memoryUsed(gpu_id_num)
            max_gpu_memUsed = np.max((max_gpu_memUsed, gpu_memUsed))
            end = time.time()
            if (end-start) >= 1:
                break
        if float(max_gpu_memUsed) < thres:
            monitor = False
            logging.info(f"Memory used: {max_gpu_memUsed} < {thres}. "
                         f"Occupying GPU[{gpu_id_num}]")
            occupy_gpu(model, thres, gpu_id_num)
        else:
            if not monitor:
                monitor = True
                logging.info(f"Memory used: {max_gpu_memUsed} >= {thres}. "
                             f"Keeping monitoring")

class SigtermException(Exception):
    pass

def sigterm_handler(signum, frame):
    logging.info("handle SIGTERM signal by raising exception")
    raise SigtermException

def main():
    import multiprocessing as mp
    mp.set_start_method('spawn')

    gpus = getGPUs()
    num_gpu = len(gpus)
    if num_gpu == 0:
        while True:
            time.sleep(10)

    ps = []
    for gpu_id_num in range(num_gpu):
        p = mp.Process(target=subprocess, args=(gpu_id_num,))
        ps.append(p)

    signal.signal(signal.SIGTERM, sigterm_handler)

    logging.info("start multiprocessing")
    for p in ps:
        p.start()

    try:
        for p in ps:
            p.join()
    except SigtermException:
        for p in ps:
            p.terminate()
    logging.info("normal exit of multiprocessing")

if __name__ == "__main__":
    main()