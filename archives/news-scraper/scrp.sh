#!/bin/bash
cd ~/Dropbox/Purdue/nlp/project1/crawler/
export PYTHONPATH=${PYTHONPATH}:~/Dropbox/Purdue/nlp/project1/crawler/

python genlinks.py
python txtcwler.py
