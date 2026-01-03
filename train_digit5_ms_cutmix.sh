conf=configs/trainers/ssl/fixmatch_ema/digit5.yaml
data_conf=configs/datasets/da/digit5.yaml 
trainer=FixMatchCutMix
opt='MODEL.BACKBONE.NAME cnn_digit5_m3sda_ms TRAINER.CUTMIX.PROB 1. TRAINER.CUTMIX.BETA 1.0  DATALOADER.NUM_WORKERS 2'
for((i=0;i<=1;i++));do
GPU=0
CUDA_VISIBLE_DEVICES=$GPU python tools/train.py --root ../../data --trainer $trainer \
	--source-domains mnist mnist_m svhn syn --target-domains usps  \
	--dataset-config-file ${data_conf} --config-file ${conf}  \
        --output-dir output/digit5_mscm/usps --resume output/adaptkernel_digit5/usps/nomodel  \
	$opt  2>&1|tee output/digit5_mscm/fm_mscm_usps_${i}.log &
((GPU=GPU+1))
CUDA_VISIBLE_DEVICES=$GPU python tools/train.py --root ../../data --trainer $trainer \
	--source-domains mnist usps svhn syn --target-domains mnist_m  \
	--dataset-config-file ${data_conf} --config-file ${conf}  \
	--output-dir output/digit5_mscm/mnist_m --resume output/adaptkernel_digit5/usps/nomodel  \
	$opt  2>&1|tee output/digit5_mscm/fm_mscm_mnistm_${i}.log &
((GPU=GPU+1))
CUDA_VISIBLE_DEVICES=$GPU python tools/train.py --root ../../data --trainer $trainer \
	--source-domains usps mnist_m svhn syn --target-domains mnist  \
	--dataset-config-file ${data_conf} --config-file ${conf}  \
	--output-dir output/digit5_mscm/mnist --resume output/adaptkernel_digit5/usps/nomodel  \
	$opt  2>&1|tee output/digit5_mscm/fm_mscm_mnist_${i}.log &
((GPU=GPU+1))
CUDA_VISIBLE_DEVICES=$GPU python tools/train.py --root ../../data --trainer $trainer \
	--source-domains mnist mnist_m usps syn --target-domains svhn  \
	--dataset-config-file ${data_conf} --config-file ${conf}  \
	--output-dir output/digit5_mscm/svhn --resume output/adaptkernel_digit5/usps/nomodel  \
	$opt  2>&1|tee output/digit5_mscm/fm_mscm_svhn_${i}.log &
((GPU=GPU+1))
CUDA_VISIBLE_DEVICES=$GPU python tools/train.py --root ../../data --trainer $trainer \
	--source-domains mnist mnist_m svhn usps --target-domains syn  \
	--dataset-config-file ${data_conf} --config-file ${conf}  \
	--output-dir output/digit5_mscm/syn --resume output/adaptkernel_digit5/usps/nomodel  \
	$opt 2>&1|tee output/digit5_mscm/fm_mscm_syn_${i}.log 
done
