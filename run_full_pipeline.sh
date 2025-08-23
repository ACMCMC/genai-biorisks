#!/bin/bash

/opt/conda/bin/conda run --live-stream --prefix /home/acreomarino/genai-biorisks/.conda python run_experiment.py ${1:-small}
/opt/conda/bin/conda run --live-stream --prefix /home/acreomarino/genai-biorisks/.conda python run_experiment.py ${1:-full}
