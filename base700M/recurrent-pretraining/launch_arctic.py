import os
import socket
import secrets
import subprocess
import argparse
):
    master_address = "$(hostname)" 
    
    return f"""
export RUN_NAME="{run_name}"
export MASTER_ADDR={master_address}
{f"export MASTER_PORT={master_port}" if master_port is not None else ""}
export WORLD_SIZE=$(( SLURM_JOB_NUM_NODES * {gpus_per_node} )) 
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

# Dirs
export OUTPUT_DIR={output_dir}
export LOGDIR=${{OUTPUT_DIR}}/logs
mkdir -p $LOGDIR

echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "Logging to $LOGDIR"
"""

def assemble_sbatch_file(
    output_dir: str,
    run_name="debug-run",
    python_invocation="train.py",
    nodes=1,
    budget_minutes=120,
    environment: str = None,
    email=None,
    gpus_per_node=1,
    partition="qGPU48", # Updated default based on user resources
    account=None,
    mem="128G", # Updated default
    cpus_per_task=16 # Updated default
):
    hours = budget_minutes // 60
    minutes = budget_minutes - hours * 60
    
    # Find a free socket
    sock = socket.socket()
    sock.bind(("", 0))
    free_socket = sock.getsockname()[1]
    sock.close()
    
    logdir = f"{output_dir}/logs"
    os.makedirs(logdir, exist_ok=True)

    # Determine GPU type request if needed (e.g. V100)
    # Defaulting to generic gpu:{N} but user can override with --extra_slurm_args if needed
    # For ARCTIC, --gres=gpu:V100:{gpus_per_node} is recommended
    gpu_request = f"gpu:V100:{gpus_per_node}" 

    sbatch_file = rf"""#!/bin/bash
#SBATCH --job-name={run_name}
#SBATCH --output={logdir}/%x_%j.log
#SBATCH --error={logdir}/%x_%j.err
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={gpus_per_node}
#SBATCH --gres={gpu_request}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
#SBATCH --time={hours}:{minutes:02d}:00
#SBATCH --partition={partition}
{f"#SBATCH --account={account}" if account else ""}
{f"#SBATCH --mail-user={email}" if email else ""}
{"#SBATCH --mail-type=FAIL,END" if email else ""}

echo $(date -u) "Preparing run..."
{load_standard_modules()}
{activate_env(environment)}
{set_generic_env_flags(run_name=run_name, gpus_per_node=gpus_per_node, master_port=free_socket, output_dir=output_dir)}

echo $(date -u) "Starting run..."
srun python -u {python_invocation}

echo $(date -u) "Job execution finished."
"""
    return sbatch_file

@dataclass
class SLURMLaunch:
    output_dir: str
    nodes: int = 1
    budget_minutes: int = 120
    environment: Optional[str] = None
    email: Optional[str] = None
    gpus_per_node: int = 1
    partition: str = "qGPU48"
    account: Optional[str] = None

    def get_output_dir(self, run_name):
        if self.output_dir is None:
            output_dir = f"{os.getcwd()}/outputs/{run_name}"
        else:
            output_dir = f"{self.output_dir}/{run_name}"
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def execute(self, python_invocation, run_name, dryrun=False):
        output_dir = self.get_output_dir(run_name)
        sbatch_content = assemble_sbatch_file(
            output_dir=output_dir,
            run_name=run_name,
            python_invocation=python_invocation,
            nodes=self.nodes,
            budget_minutes=self.budget_minutes,
            environment=self.environment,
            email=self.email,
            gpus_per_node=self.gpus_per_node,
            partition=self.partition,
            account=self.account
        )
        
        script_path = f"{output_dir}/{run_name}.slurm"
        with open(script_path, "w") as f:
            f.write(sbatch_content)
        
        print(f"Generated SLURM script at: {script_path}")
        
        if dryrun:
            print("Dryrun enabled. Script content:")
            print(sbatch_content)
        else:
            print(f"Submitting job: {script_path}")
            subprocess.run(["sbatch", script_path], check=True)

def parse_and_execute():
    parser = argparse.ArgumentParser(description="Launch training on ARCTIC HPC")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--run_name", type=str, default="default-run", help="Name of the run")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--gpus_per_node", type=int, default=1, help="GPUs per node")
    parser.add_argument("--budget_hours", type=int, default=24, help="Runtime in hours")
    parser.add_argument("--partition", type=str, default="qGPU48", help="SLURM partition")
    parser.add_argument("--account", type=str, default=None, help="SLURM account")
    parser.add_argument("--environment", type=str, default=None, help="Conda environment name/path")
    parser.add_argument("--email", type=str, default=None, help="Email for notifications")
    parser.add_argument("--dryrun", action="store_true", help="Generate script but do not submit")
    parser.add_argument("--extra_args", type=str, default="", help="Extra args for train.py")

    args = parser.parse_args()

    # Construct python invocation
    python_script = "train.py" 
    invocation = f"{python_script} "
    if args.config:
        invocation += f"--config={args.config} "
    invocation += f"--run_name={args.run_name} "
    if args.out_dir:
        invocation += f"--out_dir={args.out_dir} "
    invocation += args.extra_args

    launcher = SLURMLaunch(
        output_dir=args.out_dir,
        nodes=args.nodes,
        budget_minutes=args.budget_hours * 60,
        environment=args.environment,
        email=args.email,
        gpus_per_node=args.gpus_per_node,
        partition=args.partition,
        account=args.account
    )

    launcher.execute(invocation, args.run_name, dryrun=args.dryrun)

if __name__ == "__main__":
    parse_and_execute()
