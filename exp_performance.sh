#!/bin/bash

# Path to folder
folder_path="compressed/"

# Check if directory exists or not
if [ -d "$folder_path" ]; then
    # Get the list of files in the directory
    files=$(ls "$folder_path")

    # Run test_performance
    for file in $files; do
        python test_performance.py -weights ${folder_path}/${file}
    done
else
    echo "Directory does not exist."
fi