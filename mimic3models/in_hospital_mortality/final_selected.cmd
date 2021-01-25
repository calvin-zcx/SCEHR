python main_BCE.py --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --epochs 100 --coef_contra_loss 0  --batch_size 128 --weight_decay 0 2>&1 | tee log_BCE/final_BCE+SCL_bach_cmd_a0.bs128.weDcy0.log
python main_BCE.py --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --epochs 100 --coef_contra_loss 0.009  --batch_size 1024 --weight_decay 0 2>&1 | tee log_BCE/final_BCE+SCL_bach_cmd_a0.009.bs1024.weDcy0.log
python main_CBCE.py --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --epochs 100 --coef_contra_loss 0  --batch_size 256 --weight_decay 0 2>&1 | tee log/final_CBCE+SCL_bach_cmd_a0.bs256.weightDecay0.log
python main_CBCE.py --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --epochs 100 --coef_contra_loss 0.0025  --batch_size 256 --weight_decay 0 2>&1 | tee log/final_CBCE+SCL_bach_cmd_a0.0025.bs256.weightDecay0.log
python main_MCE.py --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda  --epochs 100 --coef_contra_loss 0  --batch_size 256 --weight_decay 0 2>&1 | tee log_MCE/final_MCE+SCL_bach_cmd_a0.bs256.weDcy0.log
python main_MCE.py --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --epochs 100 --coef_contra_loss 0.0025  --batch_size 512 --weight_decay 0 2>&1 | tee log_MCE/final_MCE+SCL_bach_cmd_a0.0025.bs512.weDcy0.log



