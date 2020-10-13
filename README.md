# covid19

## Experiments 
- [] Augment images with centered black boxes

## Using the gcloud command
The two gcloud commands available in [gcloudcompute](gcloudcompute) should be 
run in a GCP project within the cloud shell. The first command is equivalent 
to the second with one exception, the first command instantiates the VM with a 
GPU. For testing purposes it is simply cheaper to use the non-GPU option.

The gcloud command needs the `wandb` API-key to be used with the project,
so before entering the gcloud command, make sure to run in the cloud shell.
`$ APIKEY={your-wandb-api-key}`

Future variables that may change include

- `FETCH_AND_PROCESS_DATASET`: fetch the egypt dataset and process into 
tfrecords. No longer necessary if we can simply fetch the tfrecords from 
somewhere else.
