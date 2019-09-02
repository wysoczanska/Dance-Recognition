# Multimodal Dance-Recognition 

Repository for master thesis project.

Project structure:

/src

* notebooks - keeps jupyter notebooks for: data exploration, visualisation
and final multimodal fusion

* multimodal_net - scripts for visual and audio models' training and testing:
  * main_three_stream.py - trains and evaluates Three-Stream visual representation model
  
  To train:
  
  `python main_three_stream.py [path_to_letsdance_dir] --train_split_file datasets/letsdance_splits/train.csv --d Letsdance --test_split_file datasets/letsdance_splits/test.csv --lr 0.01`
  
  To evaluate: 
  
  `python main_three_stream.py [path_to_letsdance_dir] --train_split_file datasets/letsdance_splits/train.csv --d Letsdance --test_split_file datasets/letsdance_splits/val.csv --new_length 15 --print-freq 5 --lr 0.01 -b 1 --model_path ./checkpoints_inception_5d_addlayer/model_best.pth.tar --evaluate`
  
  * main_audio.py - trains and evaluates BBNN, audio representation
  
  Train:
  
  `python main_audio.py [path_to_letsdance_dir] --train_split_file datasets/letsdance_splits/train.csv --d Letsdance_audio --test_split_file datasets/letsdance_splits/test.csv --new_length 1 --new_width 216 --new_height 128  --arch BBNN` 
  
  Evaluate:
  
  `python main_audio.py [path_to_letsdance_dir] --train_split_file datasets/letsdance_splits/train.csv --d Letsdance_audio --test_split_file datasets/letsdance_splits/val.csv --new_length 1 --new_width 216 --new_height 128  --model_path ./checkpoints_bbnn/model_best.pth.tar --arch BBNN --eval ` 

To see more options type: -h


* preprocessing - audio samples download with youtube_dll, 
spectrogram extraction, 
skeleton visualization scripts