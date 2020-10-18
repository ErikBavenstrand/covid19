ECHO_PREFIX="STARTUP:"

echo "$ECHO_PREFIX Install packages"
sudo apt-get update -q
sudo apt-get install -q git wget unzip -y
#sudo apt-get install -q python3 python3-pip -y

# Get environment variables
echo "Paste your wandb API-key and press enter"
read APIKEY
GITHUB_PWD=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/GITHUB_PWD -H Metadata-Flavor:Google)
echo "Should the dataset be fetched and processed? [true/false]"
read FETCH_AND_PROCESS_DATASET

echo "$ECHO_PREFIX Using env vars APIKEY=$APIKEY,
  GITHUB_PWD=$GITHUB_PWD,
  FETCH_AND_PROCESS_DATASET=$FETCH_AND_PROCESS_DATASET,
"

echo "$ECHO_PREFIX Update pip"
curl -s https://bootstrap.pypa.io/get-pip.py | sudo python3
pip install -q --upgrade setuptools

# Required to get packages working
sudo apt-get install -q libgl1-mesa-glx -y
pip install -q six==1.13.0

# Get github repo
echo "$ECHO_PREFIX Get repo"
git clone https://carpool-master:$GITHUB_PWD@github.com/ErikBavenstrand/covid19.git
cd covid19
pip install -q -r gcloud/requirements.txt

if $FETCH_AND_PROCESS_DATASET; then
    echo "$ECHO_PREFIX Get Egypt Dataset file, can take a couple of minutes..."
    wget -q https://md-datasets-cache-zipfiles-prod.s3.eu-west-1.amazonaws.com/8h65ywd2jr-3.zip
    echo "$ECHO_PREFIX Unzipping dataset. Will probably also take a little while..."
    unzip -q 8h65ywd2jr-3.zip
    unzip -q COVID-19\ Dataset.zip
    rm 8h65ywd2jr-3.zip COVID-19\ Dataset.zip

    echo "$ECHO_PREFIX Generate the tfrecords"
    python3 generate.py
fi

wandb login $APIKEY
python3 train.py
