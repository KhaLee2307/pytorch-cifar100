#!/bin/bash

# make dir to save
if [[ -d "compression/" ]]; then
    echo "Folder already exit."
else
    mkdir compression/
fi

# compression
for i in {1..6}
do
  python compression.py -p -p_rate $((i*5))
done