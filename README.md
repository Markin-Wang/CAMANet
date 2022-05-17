# CAM-driven discriminative radiology report generation



## Prerequisites

Install the environment by command below:

```bash
conda env create --name [env_name] --file evy.yml
```

+ java jdk 1.8 is required, make sure java -h has outputs.


## Dataset
* Put the mimic-cxr and iu-xray datasets into a folder named 'data'.
* Rename (or link) the 'mimic-cxr-dsr2' to 'mimic_cxr_dsr2'.
* Put the data folder into the current project folder. The directory structure could be:
```bash
--cam_rrg
    --data
        --mimic_cxr_dsr2
        --iu_xray
```



## Training Instruction
The scripts directory contains the training scripts.

* Files in scripts/{dataset_name} are the scripts used to submit jobs in batch system.
* The training scripts are located in scripts/{dataset_name}/cfg.
* If train in the batch system such as slurm, go into the scripts/{dataset_name} directory, the run:

    `$ sbatch mimic1 (the file name)` 
* If train in the interactive session, the move the files in cfg into the scripts/{dataset_name}, or modify the files cfg by modifying the path to the main.py file,
    `$ sh run_mimic1.sh (the file name` 


