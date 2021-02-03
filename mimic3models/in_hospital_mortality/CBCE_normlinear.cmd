python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0  --batch_size 64 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.bs64.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0  --batch_size 128 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.bs128.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0  --batch_size 256 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.bs256.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 100 --coef_contra_loss 0  --batch_size 512 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.bs512.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 100 --coef_contra_loss 0  --batch_size 1024 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.bs1024.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.0005  --batch_size 64 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.0005.bs64.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.0005  --batch_size 128 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.0005.bs128.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.0005  --batch_size 256 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.0005.bs256.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 100 --coef_contra_loss 0.0005  --batch_size 512 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.0005.bs512.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 100 --coef_contra_loss 0.0005  --batch_size 1024 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.0005.bs1024.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.001  --batch_size 64 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.001.bs64.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.001  --batch_size 128 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.001.bs128.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.001  --batch_size 256 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.001.bs256.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 100 --coef_contra_loss 0.001  --batch_size 512 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.001.bs512.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 100 --coef_contra_loss 0.001  --batch_size 1024 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.001.bs1024.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.0015  --batch_size 64 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.0015.bs64.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.0015  --batch_size 128 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.0015.bs128.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.0015  --batch_size 256 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.0015.bs256.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 100 --coef_contra_loss 0.0015  --batch_size 512 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.0015.bs512.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 100 --coef_contra_loss 0.0015  --batch_size 1024 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.0015.bs1024.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.002  --batch_size 64 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.002.bs64.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.002  --batch_size 128 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.002.bs128.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.002  --batch_size 256 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.002.bs256.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 100 --coef_contra_loss 0.002  --batch_size 512 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.002.bs512.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 100 --coef_contra_loss 0.002  --batch_size 1024 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.002.bs1024.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.0025  --batch_size 64 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.0025.bs64.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.0025  --batch_size 128 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.0025.bs128.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.0025  --batch_size 256 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.0025.bs256.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 100 --coef_contra_loss 0.0025  --batch_size 512 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.0025.bs512.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 100 --coef_contra_loss 0.0025  --batch_size 1024 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.0025.bs1024.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.003  --batch_size 64 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.003.bs64.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.003  --batch_size 128 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.003.bs128.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.003  --batch_size 256 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.003.bs256.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 100 --coef_contra_loss 0.003  --batch_size 512 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.003.bs512.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 100 --coef_contra_loss 0.003  --batch_size 1024 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.003.bs1024.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.0035  --batch_size 64 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.0035.bs64.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.0035  --batch_size 128 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.0035.bs128.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.0035  --batch_size 256 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.0035.bs256.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 100 --coef_contra_loss 0.0035  --batch_size 512 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.0035.bs512.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 100 --coef_contra_loss 0.0035  --batch_size 1024 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.0035.bs1024.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.004  --batch_size 64 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.004.bs64.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.004  --batch_size 128 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.004.bs128.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.004  --batch_size 256 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.004.bs256.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 100 --coef_contra_loss 0.004  --batch_size 512 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.004.bs512.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 100 --coef_contra_loss 0.004  --batch_size 1024 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.004.bs1024.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.0045  --batch_size 64 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.0045.bs64.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.0045  --batch_size 128 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.0045.bs128.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.0045  --batch_size 256 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.0045.bs256.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 100 --coef_contra_loss 0.0045  --batch_size 512 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.0045.bs512.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 100 --coef_contra_loss 0.0045  --batch_size 1024 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.0045.bs1024.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.005  --batch_size 64 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.005.bs64.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.005  --batch_size 128 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.005.bs128.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.005  --batch_size 256 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.005.bs256.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 100 --coef_contra_loss 0.005  --batch_size 512 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.005.bs512.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 100 --coef_contra_loss 0.005  --batch_size 1024 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.005.bs1024.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.0055  --batch_size 64 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.0055.bs64.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.0055  --batch_size 128 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.0055.bs128.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.0055  --batch_size 256 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.0055.bs256.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 100 --coef_contra_loss 0.0055  --batch_size 512 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.0055.bs512.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 100 --coef_contra_loss 0.0055  --batch_size 1024 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.0055.bs1024.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.006  --batch_size 64 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.006.bs64.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.006  --batch_size 128 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.006.bs128.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.006  --batch_size 256 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.006.bs256.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 100 --coef_contra_loss 0.006  --batch_size 512 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.006.bs512.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 100 --coef_contra_loss 0.006  --batch_size 1024 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.006.bs1024.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.0065  --batch_size 64 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.0065.bs64.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.0065  --batch_size 128 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.0065.bs128.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.0065  --batch_size 256 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.0065.bs256.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 100 --coef_contra_loss 0.0065  --batch_size 512 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.0065.bs512.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 100 --coef_contra_loss 0.0065  --batch_size 1024 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.0065.bs1024.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.007  --batch_size 64 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.007.bs64.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.007  --batch_size 128 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.007.bs128.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.007  --batch_size 256 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.007.bs256.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 100 --coef_contra_loss 0.007  --batch_size 512 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.007.bs512.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 100 --coef_contra_loss 0.007  --batch_size 1024 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.007.bs1024.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.0075  --batch_size 64 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.0075.bs64.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.0075  --batch_size 128 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.0075.bs128.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.0075  --batch_size 256 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.0075.bs256.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 100 --coef_contra_loss 0.0075  --batch_size 512 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.0075.bs512.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 100 --coef_contra_loss 0.0075  --batch_size 1024 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.0075.bs1024.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.008  --batch_size 64 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.008.bs64.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.008  --batch_size 128 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.008.bs128.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.008  --batch_size 256 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.008.bs256.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 100 --coef_contra_loss 0.008  --batch_size 512 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.008.bs512.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 100 --coef_contra_loss 0.008  --batch_size 1024 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.008.bs1024.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.0085  --batch_size 64 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.0085.bs64.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.0085  --batch_size 128 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.0085.bs128.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.0085  --batch_size 256 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.0085.bs256.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 100 --coef_contra_loss 0.0085  --batch_size 512 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.0085.bs512.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 100 --coef_contra_loss 0.0085  --batch_size 1024 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.0085.bs1024.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.009  --batch_size 64 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.009.bs64.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.009  --batch_size 128 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.009.bs128.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.009  --batch_size 256 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.009.bs256.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 100 --coef_contra_loss 0.009  --batch_size 512 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.009.bs512.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 100 --coef_contra_loss 0.009  --batch_size 1024 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.009.bs1024.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.0095  --batch_size 64 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.0095.bs64.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.0095  --batch_size 128 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.0095.bs128.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.0095  --batch_size 256 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.0095.bs256.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 100 --coef_contra_loss 0.0095  --batch_size 512 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.0095.bs512.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 100 --coef_contra_loss 0.0095  --batch_size 1024 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.0095.bs1024.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.01  --batch_size 64 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.01.bs64.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.01  --batch_size 128 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.01.bs128.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 60 --coef_contra_loss 0.01  --batch_size 256 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.01.bs256.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 100 --coef_contra_loss 0.01  --batch_size 512 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.01.bs512.weightDecay0.log
python main_CBCE_norm.py --output_dir pytorch_states/CBCE_normlinear/ --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 100 --coef_contra_loss 0.01  --batch_size 1024 --weight_decay 0 2>&1 | tee log_CBCE_normlinear/CBCE+SCL_bach_cmd_a0.01.bs1024.weightDecay0.log
