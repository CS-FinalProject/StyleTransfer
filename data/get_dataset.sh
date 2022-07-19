#!/bin/bash

URL=https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/monet2photo.zip
ZIP_FILE=monet2photo.zip
TARGET_DIR=dataset
wget ${URL}
unzip ${ZIP_FILE}
rm ${ZIP_FILE}

# Adapt to project expected directory hierarchy
mkdir -p "$TARGET_DIR/train" "$TARGET_DIR/test"
mv "$TARGET_DIR/trainA" "$TARGET_DIR/train/A"
mv "$TARGET_DIR/trainB" "$TARGET_DIR/train/B"
mv "$TARGET_DIR/testA" "$TARGET_DIR/test/A"
mv "$TARGET_DIR/testB" "$TARGET_DIR/test/B"
mv "$TARGET_DIR" "dataset"
