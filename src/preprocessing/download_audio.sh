#!/bin/bash
for f in * ; do
  CLASS_PATH=/data/inputs/jersey_number_recognition/letsdance/audio/$f
  echo $CLASS_PATH
  mkdir $CLASS_PATH
  while read p; do
    youtube-dl -x --audio-format wav https://www.youtube.com/watch?v=$p -o $CLASS_PATH/$p
  done <$PWD/$f/videos.txt
done
