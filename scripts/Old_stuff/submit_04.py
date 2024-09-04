import os
import stat
import subprocess
import datetime

jobs_ids = range(1)

cwd = os.getcwd()

def make_sh_text1(seed):

    time_tag = datetime.datetime.now().strftime("%m%d")

    return f'''#!/bin/bash -l
#SBATCH -o ./job{time_tag}.out.%j.txt
#SBATCH -e ./job{time_tag}.err.%j.txt
#SBATCH -D ./
#SBATCH -J PCH_{time_tag}
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=168:00:00
#SBATCH --mail-user krohn@biochem.mpg.de     
#SBATCH --mail-type ALL

module purge
conda activate tttr

srun python /fs/pool/pool-schwille-spt/P6_FCS_HOassociation/Analysis/20240609_Fits_for_figures/scripts_as_used/FCS_Spectrum_fitting/scripts/04_parallel_fitting_PCH_param_screen.py'''

def make_sh_text2(jobscript_name):
    return f'''#!/bin/bash
    
set -e

JOBID1=$(sbatch --parsable {jobscript_name})
echo "Submitted job"
echo "    ${{JOBID1}} {jobscript_name}"
    '''

for i_job in jobs_ids:
    script_file_name = os.path.join(cwd, f'''script_{i_job}.sh''')
    # Create job script for SLURM
    open(script_file_name, 'w').write(make_sh_text1(i_job))
    # Make sure all the required rights are there
    os.chmod(script_file_name, stat.S_IRWXU)
    
    submission_file_name = os.path.join(cwd, f'''submit_{i_job}.sh''')
    # Create wrapper script for SLURM
    open(submission_file_name, 'w').write(make_sh_text2(script_file_name))
    # Make sure all the required rights are there
    os.chmod(submission_file_name, stat.S_IRWXU)

    # Submit job script
    subprocess.call(submission_file_name) 
    # os.system(f'''sbatch script_{i_job}.sh''') # Old version that did not seem to work well