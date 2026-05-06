This folder holds the training harness and helper scripts to pretrain bert and mcberth models on temporally annoted text data.

Requirements.txt should hold the right packages needed to run any of the pretrainings locally. This works for testing purposes and insuring the scripts will actually run. To run any of the scripts that use google cloud you will need to get a service account .json file. This is your credentials that lets you access the google cloud API. Ask Professor for the priviledges needed to download this file or have another researcher send it to you. Then make sure it is named "nlp-research-sp26-8499634f1c62.json" this isn't crucial but it's what all the scripts expect.

To train on the full dataset it's better to use the docker version of the training scripts in mcBERTh_training/ .

If you haven't used docker before watch this tutorial before messing with anything: https://www.youtube.com/watch?v=Ud7Npgi6x8E
 - this gives this basics of what it is we are trying to do with docker and details how to install it
 - to summarize: we need to run our scripts on google's computer. A docker container is like an extra powerful virtual environment that tells google's computer not only what dependencies to install but also how to run the program. It's basically a set of instructions for the google trainer computer.

You will also need to use the google cloud services command line interface (GCS CLI) to authenticate your local docker client. Follow this tutorial to do that: https://docs.cloud.google.com/sdk/docs/install-sdk 

Once your docker is installed and authenticated, we can build, push, and run a training. 

To build the docker image use the following commands from inside the mcBERTh_training/ folder:

1. docker image build . -t us-central1-docker.pkg.dev/nlp-research-sp26/mcberth-training/mcberth-training:test_mcberth

2. docker push us-docker.pkg.dev/nlp-research-sp26/mcberth-training/mcberth-training:latest 

Command one builds the files into a docker image on your local device using the Docker Desktop application. Command two pushes the built image to the specific google cloud URI in the artifact registry. Once an image with the same name and tag (ex. image_name:image_tag) shows up in the artifact registry you can start the training.

submit_job.py tells google to run a training using a specific docker image, on specific hardware, using specific parameters, and then save the output to another URI. If it submits properly it the command line will show status 3. Once this happens you can stop the script on your device and observe progress from the google cloud training logs. 

If the script won't run make sure you are running it from inside the mcBERTh_training/ folder and that you have put a copy of "nlp-research-sp26-8499634f1c62.json" in the same folder.