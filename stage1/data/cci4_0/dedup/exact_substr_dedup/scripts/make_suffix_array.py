
# Copyright 2021 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import time
import argparse
import multiprocessing as mp
import numpy as np
from loguru import logger
from utils import human_readable_size


def make_suffix_array(big_sequence_path, tmp_path):
    data_size = os.path.getsize(big_sequence_path)

    logger.info(f".big_sequence file size: {human_readable_size(data_size)}")

    HACK = 100000

    started = []

    if data_size > 10e9:
        total_jobs = 100
        jobs_at_once = 20
    elif data_size > 1e9:
        total_jobs = 96
        jobs_at_once = 96
    elif data_size > 10e6:
        total_jobs = 4
        jobs_at_once = 4
    else:
        total_jobs = 1
        jobs_at_once = 1

    S = data_size//total_jobs

    logger.info(f"total_jobs: {total_jobs}, jobs_at_once: {jobs_at_once}")

    for jobstart in range(0, total_jobs, jobs_at_once):
        wait = []
        for i in range(jobstart, jobstart + jobs_at_once):
            s, e = i*S, min((i+1)*S+HACK, data_size)
            # cmd = "./target/debug/dedup_dataset make-part --data-file %s --start-byte %d --end-byte %d"%(big_sequence_path, s, e)
            cmd = f"./target/debug/dedup_dataset make-part --data-file {big_sequence_path} --start-byte {s} --end-byte {e}"
            started.append((s, e))
            print(cmd)
            wait.append(os.popen(cmd))
            
            if e == data_size:
                break

        logger.info("Waiting for jobs to finish")
        [x.read() for x in wait]

    logger.info("Checking all wrote correctly")

    while True:
        files = ["%s.part.%d-%d"%(big_sequence_path,s, e) for s,e in started]
        
        wait = []
        for x,(s,e) in zip(files,started):
            go = False
            if not os.path.exists(x):
                print("GOGO")
                go = True
            else:
                size_data = os.path.getsize(x)
                FACT = np.ceil(np.log(size_data)/np.log(2)/8)
                print("FACT", FACT,size_data*FACT, os.path.getsize(x+".table.bin"))
                if not os.path.exists(x) or not os.path.exists(x+".table.bin") or os.path.getsize(x+".table.bin") == 0 or size_data*FACT != os.path.getsize(x+".table.bin"):
                    go = True
            if go:
                cmd = "./target/debug/dedup_dataset make-part --data-file %s --start-byte %d --end-byte %d"%(big_sequence_path, s, e)
                print(cmd)
                wait.append(os.popen(cmd))
                if len(wait) >= jobs_at_once:
                    break
        logger.warning("Rerunning", len(wait), "jobs because they failed.")
        [x.read() for x in wait]
        time.sleep(1)
        if len(wait) == 0:
            break

    logger.info("Merging suffix trees")

    # os.makedirs("./tmp", exist_ok=True)
    os.makedirs(tmp_path, exist_ok=True)
    # os.popen(f"rm {tmp_path}/out.table.bin.*").read()

    torun = " --suffix-path ".join(files)
    # print("./target/debug/dedup_dataset merge --output-file %s --suffix-path %s --num-threads %d"%(tmp/out.table.bin", torun, mp.cpu_count()))
    print(f"./target/debug/dedup_dataset merge --output-file {tmp_path}/out.table.bin --suffix-path {torun} --num-threads {mp.cpu_count()}")
    # pipe = os.popen("./target/debug/dedup_dataset merge --output-file %s --suffix-path %s --num-threads %d"%("tmp/out.table.bin", torun, mp.cpu_count()))
    pipe = os.popen(f"./target/debug/dedup_dataset merge --output-file {tmp_path}/out.table.bin --suffix-path {torun} --num-threads {mp.cpu_count()}")
    output = pipe.read()
    if pipe.close() is not None:
        logger.critical("Something went wrong with merging.")
        logger.critical("Please check that you ran with ulimit -Sn 100000")
        exit(1)
    #exit(0)
    logger.info("Now merging individual tables")
    # os.popen("cat tmp/out.table.bin.* > tmp/out.table.bin").read()
    os.popen(f"cat {tmp_path}/out.table.bin.* > {tmp_path}/out.table.bin").read()
    logger.info("Cleaning up")
    # os.popen("mv tmp/out.table.bin %s.table.bin"%big_sequence_path).read()
    os.popen(f"mv {tmp_path}/out.table.bin {big_sequence_path}.table.bin").read()

    if os.path.exists(big_sequence_path+".table.bin"):
        if os.path.getsize(big_sequence_path+".table.bin")%os.path.getsize(big_sequence_path) != 0:
            logger.critical("File size is wrong")
            exit(1)
    else:
        logger.critical("Failed to create table")
        exit(1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--working-folder", type=str, default="./")
    
    args = parser.parse_args()

    make_suffix_array(
        big_sequence_path=os.path.join(args.working_folder, "es/dataset.big_sequence"),
        tmp_path=os.path.join(args.working_folder, "tmp")
    )
