echo "Source this file, not execute"
echo "Make sure to use single virtual environment (not to start a Python venv withiin Conda venv"
DIR=$(realpath .)
source ./venv/bin/activate
export PATH="${DIR}/venv/bin:${PATH}"
