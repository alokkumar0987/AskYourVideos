# Dot-source this in a PowerShell window before using askvideo:
#   . .\env.ps1
# Keeps every download and cache on this drive (models, pip, temp files) and
# activates the project's virtual environment.

$root = $PSScriptRoot
$env:HF_HOME = "$root\.cache\hf"
$env:TORCH_HOME = "$root\.cache\torch"
$env:PIP_CACHE_DIR = "$root\.cache\pip"
$env:TEMP = "$root\.cache\tmp"
$env:TMP = "$root\.cache\tmp"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTHONWARNINGS = "ignore::FutureWarning"
& "$root\.venv\Scripts\Activate.ps1"
Write-Host "askvideo ready (caches on $root\.cache)"
