#! /bin/bash

cd ./data
tar_files=$(find ./ -type f -name "*.tar")

for tar_file in $tar_files; do
    tar -xf $tar_file
done

cd ..