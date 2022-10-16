import os
import shutil

DF_DATASET_DIR = "./manipulated/"
REAL_DATASET_DIR = "./real/"
TARGET_DIR = "./testing_dataset"

def dataset_dist():
    
    benchmarks = [5, 50, 250, 500]
    for num in range(len(benchmarks)):
        count = 0
        for file in os.listdir(DF_DATASET_DIR):
            if count < benchmarks[num]:
                original = f'{DF_DATASET_DIR}/{file}'
                target = f'{TARGET_DIR}/{benchmarks[num] * 2}/df/{file}'
                shutil.copyfile(original, target)
                count += 1

    for num in range(len(benchmarks)):
        count = 0
        for file in os.listdir(REAL_DATASET_DIR):
            if count < benchmarks[num]:
                original = f'{REAL_DATASET_DIR}/{file}'
                target = f'{TARGET_DIR}/{benchmarks[num] * 2}/real/{file}'
                shutil.copyfile(original, target)
                count += 1

if __name__ == "__main__":
    dataset_dist()