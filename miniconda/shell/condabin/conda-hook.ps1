$Env:CONDA_EXE = "/mnt/2w12-data/audio-sampler-v2/miniconda/bin/conda"
$Env:_CE_M = $null
$Env:_CE_CONDA = $null
$Env:_CONDA_ROOT = "/mnt/2w12-data/audio-sampler-v2/miniconda"
$Env:_CONDA_EXE = "/mnt/2w12-data/audio-sampler-v2/miniconda/bin/conda"
$CondaModuleArgs = @{ChangePs1 = $True}
Import-Module "$Env:_CONDA_ROOT\shell\condabin\Conda.psm1" -ArgumentList $CondaModuleArgs

Remove-Variable CondaModuleArgs