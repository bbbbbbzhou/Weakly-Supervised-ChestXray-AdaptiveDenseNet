python -W ignore test.py \
--resume './outputs/train_ChestXray14_densenetADA/checkpoints/model_best.pt' \
--experiment_name 'test_ChestXray14_densenetADA' \
--model_type 'model_wsl' \
--data_root '../Data/ChestXray14/' \
--net_G 'densenetADA' \
--gpu_ids 0

