POST_SCRIPT_URL=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/POST_SCRIPT_URL -H Metadata-Flavor:Google)

wget $POST_SCRIPT_URL
chmod 777 post_script.sh
./post_script.sh
