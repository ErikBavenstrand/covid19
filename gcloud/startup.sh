ECHO_PREFIX="STARTUP:"

echo "$ECHO_PREFIX Install packages"
sudo apt-get update -q
sudo apt-get install -q git wget unzip -y
sudo apt-get install -q python python3-pip -y

# Get environment variables
APIKEY=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/APIKEY -H Metadata-Flavor:Google)
GITHUB_PWD=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/GITHUB_PWD -H Metadata-Flavor:Google)
FETCH_AND_PROCESS_DATASET=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/FETCH_AND_PROCESS_DATASET -H Metadata-Flavor:Google)
ADD_GPU_DRIVER=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/ADD_GPU_DRIVER -H Metadata-Flavor:Google)

echo "$ECHO_PREFIX Using env vars APIKEY=$APIKEY,
  GITHUB_PWD=$GITHUB_PWD,
  FETCH_AND_PROCESS_DATASET=$FETCH_AND_PROCESS_DATASET,
  ADD_GPU_DRIVER=$ADD_GPU_DRIVER
"


if $ADD_GPU_DRIVER; then
    echo "$ECHO_PREFIX Add the cuda repo"
    # Add NVIDIA package repositories
    sudo wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.1.243-1_amd64.deb
    sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
    sudo dpkg -i cuda-repo-ubuntu1804_10.1.243-1_amd64.deb
    sudo apt-get update -q
    sudo wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
    sudo apt install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb -y
    sudo apt-get update -q

    # Install NVIDIA driver
    sudo apt-get install -y --no-install-recommends nvidia-driver-450
    # Reboot. Check that GPUs are visible using the command: nvidia-smi

    # Install development and runtime libraries (~4GB)
    sudo apt-get install -y --no-install-recommends \
        cuda-10-1 \
        libcudnn7=7.6.5.32-1+cuda10.1  \
        libcudnn7-dev=7.6.5.32-1+cuda10.1

    # Install TensorRT. Requires that libcudnn7 is installed above.
    sudo apt-get install -y --no-install-recommends libnvinfer6=6.0.1-1+cuda10.1 \
        libnvinfer-dev=6.0.1-1+cuda10.1 \
        libnvinfer-plugin6=6.0.1-1+cuda10.1
    # Verify gpu-driver installation
    nvidia-smi
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.2/lib64/:/usr/local/cuda-10.1/lib64/
fi

echo "$ECHO_PREFIX Update pip"
curl -s https://bootstrap.pypa.io/get-pip.py | sudo python3
sudo pip install -q --upgrade setuptools

# Required to get packages working
sudo apt-get install -q libgl1-mesa-glx -y
sudo pip install -q six==1.13.0

# Get github repo
echo "$ECHO_PREFIX Get repo"
git clone https://carpool-master:$GITHUB_PWD@github.com/ErikBavenstrand/covid19.git
cd covid19
sudo pip install -q -r requirements.txt

if $FETCH_AND_PROCESS_DATASET; then
    echo "$ECHO_PREFIX Get Egypt Dataset file, can take a couple of minutes..."
    wget -q https://md-datasets-cache-zipfiles-prod.s3.eu-west-1.amazonaws.com/8h65ywd2jr-3.zip
    echo "$ECHO_PREFIX Unzipping dataset. Will probably also take a little while..."
    sudo unzip -q 8h65ywd2jr-3.zip
    sudo unzip -q COVID-19\ Dataset.zip
    sudo rm 8h65ywd2jr-3.zip COVID-19\ Dataset.zip

    echo "$ECHO_PREFIX Generate the tfrecords"
    sudo python3 generate.py
fi

sudo wandb login $APIKEY
sudo python3 train.py
