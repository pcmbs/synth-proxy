#!/bin/bash


#SBATCH -J sp                                 # job name
#SBATCH -D /home/users/p/paolo_combes_1       # batch script's working dir
#SBATCH --open-mode=append					  # append to slurm output if already exists

#SBATCH --ntasks=1                            # number of task to run
#SBATCH --gres=gpu:tesla:1					  # consumable resource
#SBATCH --cpus-per-task=16                    # number of CPU required per task
#SBATCH --mem=32G                             # memory requested

# Expected Runtime in D-HH:MM:SS (Max Walltime)
#SBATCH --time=0-48:00:00

# Partition to submit to
#SBATCH -p gpu

# Job Status by E-mail
#SBATCH --mail-type=ALL
#SBATCH --mail-user=paolocombes@gmail.com

# Send signal to interrupt job before Walltime
#SBATCH --signal=B:USR1@600


# Name of the singularity container
SINGULARITY_CONTAINER="synth-proxy_hpc.sif" 

# Get the job name, args passed to the script, and requested time limit
JOB_NAME=$SLURM_JOB_NAME
ARGS="$@"
TIME_LIMIT=$(squeue -j $SLURM_JOB_ID -h --Format TimeLimit)

# Function to gracefully handle shutdown
graceful_shutdown() {
    echo "USR1 signal received. Attempting graceful shutdown..."

    RUN_INFO="$(pwd)/mounts/logs/$SLURM_JOB_ID.sh"

    # Check if $RUN_INFO exists and source env vars from it to resubmit the job
    if [ -f "$RUN_INFO" ]; then
        echo "Sourcing updated environment variables from $RUN_INFO"
        source "$RUN_INFO"
        
        # Display the updated values
        echo "WANDB_RUN_ID: $WANDB_RUN_ID"
        echo "HYDRA_RUN_DIR: $HYDRA_RUN_DIR"

        # Add environment variables to the ARGS
        ARGS_RESUME="$ARGS ckpt_path=last run_id=$WANDB_RUN_ID hydra.run.dir=$HYDRA_RUN_DIR"

        # Resubmit the job with the updated arguments (check time limit)
        echo "Resubmitting job with updated arguments: $ARGS_RESUME"
        sbatch --job-name="$JOB_NAME" $0 "$ARGS_RESUME"

        # Remove the file after sourcing the environment variables
        rm -f "$RUN_INFO"
        echo "Job resubmitted successfully."
    
    else
        echo "$RUN_INFO not found, auto-requeue disabled."
    fi
	
	# Wait for the srun process to finish before proceeding
    echo "Waiting for current srun process (PID: $SRUN_PID) to finish..."
    wait $SRUN_PID
}

# Trap the SIGUSR1 signal to initiate graceful shutdown
trap 'graceful_shutdown' usr1

# checking for singularity container
if [ -f "$SINGULARITY_CONTAINER" ]; then
	echo "found candidate sif file for singularity runs as $(pwd)/$SINGULARITY_CONTAINER"
else
	echo "didn't find $SINGULARITY_CONTAINER in the codebase folder. please be sure that the sif filename matches and is copied into the gateway path: $(pwd)"
	exit 1
fi

# Showing experiment details
echo "Starting experiment..."
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "TIME_LIMIT: $TIME_LIMIT"
echo "PWD: $(pwd)"
echo "SINGULARITY_CONTAINER: $SINGULARITY_CONTAINER"
echo "JOB_NAME: $SLURM_JOB_NAME"
echo "ARGS: $ARGS"


# Load singularity
echo "Loading singularity..."
module load singularity/4.0.2
singularity version

# Run Experiment
echo "Running Experiment..."
srun singularity exec --cleanenv --pid --nv --cwd /workspace --env SLURM_JOB_ID="$SLURM_JOB_ID" \
	--env-file $(pwd)/mounts/.env \
	-B $(pwd)/mounts/checkpoints:/workspace/checkpoints \
	-B $(pwd)/mounts/configs:/workspace/configs \
	-B $(pwd)/mounts/data:/workspace/data \
	-B $(pwd)/mounts/logs:/workspace/logs \
	-B $(pwd)/mounts/src:/workspace/src \
	-B $(pwd)/mounts/tests:/workspace/tests \
	-B $(pwd)/mounts/pyproject.toml:/workspace/pyproject.toml \
	$(pwd)/$SINGULARITY_CONTAINER bash -c "pip install --user -e . && $ARGS" &

# Store the PID of the background process
SRUN_PID=$!

# Wait for the background process (srun) to finish
wait $SRUN_PID

# Exit with success
echo "Job finished successfully!"
exit