# Jet Tagging Classification Project

## Introduction

This project is my Jet Tagging project assignment related to the Physics Application of Ai course at the university of Geneva. A paper about the project is found at the root folder of this repo

## Structure

The stracture was inspierd by the project template project given by the course team.

## Running the project

To run the project, you should first download the dataset with the instructions mentioned at:
https://gitlab.unige.ch/ai-in-physics-course/projects/jettagging

then, install the required packages:

if you are using conda (prefered):
```
conda env create -f environment.yml
```

if not, install the packages using pip:
```
 pip install -r requirements.txt
```

you can run the configuration that the paper with:

```
    bash run.sh
```

## Additional Information

There is simple code improvement that I have not Implemented yet. The main ones are:
* formatting the code using black and flake8
* make the eval script read all its parameters for the model yaml file (now you need to transfer to the eval if you are using multi-class classification or binary one, but it should infer it automatically)

The code contains some extra features that was useful for development and debugging, but not used for the final paper. I decide to keep that code and I hope it will not confuse anyone who use this repo!

