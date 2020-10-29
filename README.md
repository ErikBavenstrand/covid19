# covid19

## Experiments 
- [] Augment images with centered black boxes

## Using the gcloud command
The two gcloud commands available in [gcloudcompute](/gcloud/gcloudcompute) should be 
run in a GCP project within the **cloud shell**. The first command is equivalent 
to the second, with one exception, the first command instantiates the VM with a 
GPU. For testing purposes it is simply cheaper to use the non-GPU option.

Once the gcloud command has been executed, do the following

1. SSH into the the machine
1. Execute the commands below (found in [prep.sh](/gcloud/prep.sh)
1. The scipt will prompt for your `wandb` API-key and whether the dataset 
(currently only mendeley supported) should be fetched. False currently expects
the tf-records to exist on the machine.

### Prep.sh
```
POST_SCRIPT_URL=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/POST_SCRIPT_URL -H Metadata-Flavor:Google)

wget $POST_SCRIPT_URL
chmod 777 post-script.sh
./post-script.sh
```
