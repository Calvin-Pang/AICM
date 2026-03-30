# AICM


This repository contains the official implementation for MPR-Diff introduced in the following paper:

[ISBI 2026 Oral] AICM: An Anatomical-Prior Integrated Conditional Consistency Model for Self-Supervised MRI  Through-Plane Super-Resolution
<br>
[Kaifeng Pang](https://kfpang.com), [Qi Miao](https://mrrl.ucla.edu/people/qi-miao-md-phd), [Alex Ling Yu Hung](https://web.cs.ucla.edu/~alexhung/), [Changsuk Oh]([https://kaizhao.net/](https://mrrl.ucla.edu/people/changsuk-oh-phd)), [Kai Zhao](https://kaizhao.net/), [Qiudi He](https://mrrl.ucla.edu/people/qiudi-he), Marcel Dominik Nickel, Fei Han, [Kyunghyun Sung](https://mrrl.ucla.edu/people/kyung-sung-phd)
<br>
IEEE International Symposium on Biomedical Imaging (ISBI) 2026

### Training

```
mpiexec -n 1 python -m scripts.cm_train_mri --training_mode consistency_training --target_ema_mode improved --start_ema 0.95 --scale_mode ladder --start_scales 10 --end_scales 1280 --total_training_steps 800000 --loss_norm PH-l2 --lr_anneal_steps 0 --dropout 0.2 --teacher_dropout 0.2 --ema_rate 0.9999,0.99994,0.9999432189950708 --global_batch_size 4 --image_size 320 --lr 0.00005 --num_channels 64 --num_head_channels 64 --num_res_blocks 1 --resblock_updown True --schedule_sampler uniform --use_fp16 True --weight_decay 0.0 --weight_schedule uniform --epoch 260 --save_interval 50000 --log_interval 500 --save_dir PATH_TO_SAVE_DIR --attention_resolutions 256,128,64,32,16 --channel_mult 1,2,4,8,16 --scan cor --scale 5.76 --crop 288 --ds sp --input_mode stack --fm PATH_TO_PRETRAINED_MEDSAM_CHECKPOINT
```
