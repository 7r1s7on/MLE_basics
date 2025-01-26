# MLE basics
This project is a homework for getting familiar with git, github and docker.

## Project structure

```
MLE_basic_example
├── data                      # Data files used for training and inference (it can be generated with data_generation.py script)
│   ├── train.csv
│   └── inference.csv
├── data_process              # Scripts used for data processing and generation
│   ├── data_processing.py
│   └── __init__.py           
├── inference                 # Scripts and Dockerfiles used for inference
│   ├── Dockerfile
│   ├── run.py
│   └── __init__.py
├── models                    # Folder where trained models are stored
│   └── various model files
├── training                  # Scripts and Dockerfiles used for training
│   ├── Dockerfile
│   ├── train.py
│   └── __init__.py
├── utils.py                  # Utility functions and classes that are used in scripts
├── settings.json             # All configurable parameters and settings
└── README.md
```

## Prerequisites
python, git, docker

To run this project, take following steps:
1. Clone the repository.
2. Create a `.env` file inside the project folder, and write `CONF_PATH=settings.json` in it.
3. Run `data_processing.py` script in data_process folder. This will download and split the data into training and inference datasets.
4. Run `unittests.py` in unittests directory to ensure the dataset is divided properly.
5. Open git bash and make sure it is located at root of the project folder. Also docker should be running.
6. To train the model, write `docker build -f ./training/Dockerfile --build-arg settings_name=settings.json -t training_image .` in terminal and execute it. This will train and save the model inside docker container.
7. Then run the container with following command `docker run -dit training_image`.
8. Copy a model from container to your local machine with `docker cp <container_id>:/app/models/model.pth ./models` command. `<container_id>` should be replaced with id of your container.
9. To make inference, write `docker build -f ./inference/Dockerfile --build-arg model_name=model.pth --build-arg settings_name=settings.json -t inference_image .` command and execute it.
10. Run the inference container with `docker run -dit inference_image`.

After all these steps, you should have predictions.csv in results folder.