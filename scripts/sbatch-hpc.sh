#!/bin/bash


#SBATCH -J synth_proxy                        # job name
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


# Whether to requeue the job if it reaches max walltime (optional)
# job can be requeue by passing the "-r" flag
REQUEUE_JOB=0 # don't requeue by default

# Name of the singularity container and instance to run
# instance can be changed by passing the "-i" flag
SINGULARITY_CONTAINER="synth-proxy_hpc.sif" 
SINGULARITY_INSTANCE="synth-proxy" 

# Parse arguments
while getopts ":rti:" opt; do
  case ${opt} in
    r )
      REQUEUE_JOB=1
      ;;
    i )
      SINGULARITY_INSTANCE="$OPTARG"
      ;;
    \? )
      echo "Usage: $(basename $0) [-r] [-t] [-i singularity-instance] args"
	  exit 1
      ;;
  esac
done
shift "$((OPTIND -1))" # remove processed arguments

# instance args
ARGS="$@"

# Function to gracefully handle shutdown
graceful_shutdown() {
	echo "Signal received. Initiating graceful shutdown..."
	# Get PID from Python main process running in the container
	python_pid=$(singularity exec instance://$SINGULARITY_INSTANCE ps -eo pid,cmd | grep "$ARGS" | grep -v grep | awk '{print $1}' | head -n 1)

	# Sending USR1 signal to python main process
	echo "Received shutdown signal. Attempting graceful shutdown of the Python process with PID $python_pid..."
	singularity exec instance://$SINGULARITY_INSTANCE kill -USR1 $python_pid

	# Wait for all subprocesses/job-steps to finish
	wait

	# Stopping singularity instance
	echo "Stopping Singularity instance..."
	singularity instance stop $SINGULARITY_INSTANCE

	# Requeue job if necessary
	if [ $REQUEUE_JOB == 1 ]; then
		echo "Requeueing job..."
		scontrol requeue $SLURM_JOBID
	fi

	# Exit with success
	echo "Job interruputed successfully!"
	exit 0
}

# Trap the SIGUSR1 signal to initiate graceful shutdown
trap 'graceful_shutdown' USR1 SIGINT SIGTERM

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
echo "PWD: $(pwd)"
echo "SINGULARITY_CONTAINER: $SINGULARITY_CONTAINER"
echo "SINGULARITY_INSTANCE: $SINGULARITY_INSTANCE"
echo "ARGS: $ARGS"
echo "REQUEUE_JOB: $REQUEUE_JOB"
echo "RUN_TESTS: $RUN_TESTS"

# Load singularity
echo "Loading singularity..."
module load singularity/4.0.2
singularity version

# Run Experiment
echo "Running Experiment..."
srun singularity exec --cleanenv --pid --nv --cwd /workspace \
	--env-file $(pwd)/mounts/.env \
	-B $(pwd)/mounts/checkpoints:/workspace/checkpoints \
	-B $(pwd)/mounts/configs:/workspace/configs \
	-B $(pwd)/mounts/datasets:/workspace/data/datasets \
	-B $(pwd)/mounts/logs:/workspace/logs \
	-B $(pwd)/mounts/src:/workspace/src \
	-B $(pwd)/mounts/tests:/workspace/tests \
	$(pwd)/$SINGULARITY_CONTAINER bash -c "pip install --user -e . && $ARGS" &
wait

# Exit with success
echo "Job finished successfully!"
exit