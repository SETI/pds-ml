# This sets up a PDS development environment.

#*************************************************************************************************************
# Determine if we are running on the NAS
# 'hostname -d' is not available on MAC OS so only do this query if not on a darwin $OSTYPE
export isOnNas=False
if ! [[ $OSTYPE =~ darwin* ]] ; then
    if hostname -d | grep -q 'nas.nasa.gov' ; then
        # If this is a noninteractive shell, then we need to do some 
        # additional setup.
        if [[ $- == *i* ]]; then
            source /usr/local/lib/global.profile
        fi
        export isOnNas=True
    fi
fi


#*************************************************************************************************************
# If on the NAS then setup GPU node shortcuts
if [ "${isOnNas}" != "true" ]; then
    # Specify the PDS group account
    export GROUP=s2572

    # Interactive GPU nodes
    # This is one v100 on a skylake GPU node, it uses 1/4 of the resources on the node
    alias gpu_node_sky_1='qsub -I -W group_list=$GROUP -q v100@pbspl4 -l select=1:model=sky_gpu:ngpus=1:ncpus=9:mem=96g,place=vscatter:shared,walltime=8:00:00'
    # This uses 1/2 of the resources (2 v100 GPUS) of a skylake GPU node
    alias gpu_node_sky_2='qsub -I -W group_list=$GROUP -q v100@pbspl4 -l select=1:model=sky_gpu:ngpus=2:ncpus=18:mem=192g,place=vscatter:shared,walltime=8:00:00'
    
    # 1/4 of a cascadelake GPU node (1 v100 GPU)
    alias gpu_node_cas_1='qsub -I -W group_list=$GROUP -q v100@pbspl4 -l select=1:model=cas_gpu:ngpus=1:ncpus=12:mem=96g,place=vscatter:shared,walltime=8:00:00'
    # 1/2 of a cascade loake node (2 v100 GPUs)
    alias gpu_node_cas_2='qsub -I -W group_list=$GROUP -q v100@pbspl4 -l select=1:model=cas_gpu:ngpus=2:ncpus=24:mem=192g,place=vscatter:shared,walltime=8:00:00'
fi

#*************************************************************************************************************
# Conda environment

# Conda environments
export CONDA_ENVS_PATH=/nobackupp15/jcsmit20/.conda/envs

# conda path
export CONDA_ROOT_PATH=/nobackupp15/jcsmit20/miniconda3

# Need to point the C shared library path correctly
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_ROOT_PATH/lib

# >>> conda initialize >>>
__conda_setup="$('$CONDA_ROOT_PATH/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "$CONDA_ROOT_PATH/etc/profile.d/conda.sh" ]; then
        . "$CONDA_ROOT_PATH/etc/profile.d/conda.sh"
    else
        export PATH="$CONDA_ROOT_PATH/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

# Activate pds-env conda environment
conda activate pds-env
