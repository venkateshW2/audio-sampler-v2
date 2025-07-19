export MAMBA_ROOT_PREFIX="/mnt/2w12-data/audio-sampler-v2/miniconda"
__mamba_setup="$("/mnt/2w12-data/audio-sampler-v2/miniconda/bin/mamba" shell hook --shell posix 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__mamba_setup"
else
    alias mamba="/mnt/2w12-data/audio-sampler-v2/miniconda/bin/mamba"  # Fallback on help from mamba activate
fi
unset __mamba_setup
