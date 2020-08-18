python -W ignore train.py \
--experiment_name 'train_ChestXray14_densenetADA' \
--model_type 'model_wsl' \
--dataset 'ChestXray14' \
--data_root '../Data/ChestXray14/' \
--net_G 'densenetADA' \
--n_class 14 \
--batch_size 36 \
--lr 1e-4 \
--eval_epochs 4 \
--save_epochs 4 \
--snapshot_epochs 4 \
--AUG \
--gpu_ids 0

