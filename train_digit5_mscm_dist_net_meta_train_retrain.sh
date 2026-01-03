conf=configs/trainers/ssl/fixmatch_ema/digit5.yaml
data_conf=configs/datasets/da/digit5.yaml 
trainer=FixMatchMSCMDistNetMetaLearnRetrain
opt='MODEL.BACKBONE.NAME cnn_digit5_m3sda_ms TRAINER.RETRAIN.RATIO 1. TRAINER.CUTMIX.PROB 1. TRAINER.RETRAIN.EPOCH 20 OPTIM.MAX_EPOCH 50'
a=1
for((i=0;i<=4;i++));do
GPU=0
opt='MODEL.BACKBONE.NAME cnn_digit5_m3sda_ms TRAINER.RETRAIN.RATIO 1. TRAINER.CUTMIX.PROB 1. TRAINER.RETRAIN.EPOCH 20 OPTIM.MAX_EPOCH 20 TRAINER.METALEARN.TYPE sgd TRAINER.METALEARN.LR 0.00005 TRAINER.METALEARN.STEP 30 MODEL.INIT_WEIGHTS output/mscm_dist_net_retrain_digit5/usps/model_tar/model.pth.tar-50'
CUDA_VISIBLE_DEVICES=$GPU python tools/train.py --root ../../data --trainer $trainer \
	--source-domains mnist mnist_m svhn syn --target-domains usps  \
	--dataset-config-file ${data_conf} --config-file ${conf}  \
        --output-dir output/mscm_dist_net_meta_learn_retrain_digit5/usps --resume output/adaptkernel_digit5/usps/nomodel  \
	$opt 2>&1|tee output/mscm_dist_net_meta_learn_retrain_digit5/fm_mscm_retrain_usps_${i}.log &
((GPU=GPU+1))
opt='MODEL.BACKBONE.NAME cnn_digit5_m3sda_ms TRAINER.RETRAIN.RATIO 1. TRAINER.CUTMIX.PROB 1. TRAINER.RETRAIN.EPOCH 20 OPTIM.MAX_EPOCH 20 TRAINER.METALEARN.TYPE sgd TRAINER.METALEARN.LR 0.00005 TRAINER.METALEARN.STEP 30 MODEL.INIT_WEIGHTS output/mscm_dist_net_retrain_digit5/mnist_m/model_tar/model.pth.tar-50'
CUDA_VISIBLE_DEVICES=$GPU python tools/train.py --root ../../data --trainer $trainer \
	--source-domains mnist usps svhn syn --target-domains mnist_m  \
	--dataset-config-file ${data_conf} --config-file ${conf}  \
	--output-dir output/mscm_dist_net_meta_learn_retrain_digit5/mnist_m --resume output/adaptkernel_digit5/usps/nomodel  \
	$opt 2>&1|tee output/mscm_dist_net_meta_learn_retrain_digit5/fm_mscm_retrain_mnistm_${i}.log &
((GPU=GPU+1))
opt='MODEL.BACKBONE.NAME cnn_digit5_m3sda_ms TRAINER.RETRAIN.RATIO 1. TRAINER.CUTMIX.PROB 1. TRAINER.RETRAIN.EPOCH 20 OPTIM.MAX_EPOCH 20 TRAINER.METALEARN.TYPE sgd TRAINER.METALEARN.LR 0.00005 TRAINER.METALEARN.STEP 30 MODEL.INIT_WEIGHTS output/mscm_dist_net_retrain_digit5/mnist/model_tar/model.pth.tar-50'
CUDA_VISIBLE_DEVICES=$GPU python tools/train.py --root ../../data --trainer $trainer \
	--source-domains usps mnist_m svhn syn --target-domains mnist  \
	--dataset-config-file ${data_conf} --config-file ${conf}  \
	--output-dir output/mscm_dist_net_meta_learn_retrain_digit5/mnist --resume output/adaptkernel_digit5/usps/nomodel  \
	$opt 2>&1|tee output/mscm_dist_net_meta_learn_retrain_digit5/fm_mscm_retrain_mnist_${i}.log &
((GPU=GPU+1))
opt='MODEL.BACKBONE.NAME cnn_digit5_m3sda_ms TRAINER.RETRAIN.RATIO 1. TRAINER.CUTMIX.PROB 1. TRAINER.RETRAIN.EPOCH 20 OPTIM.MAX_EPOCH 20 TRAINER.METALEARN.TYPE sgd TRAINER.METALEARN.LR 0.00005 TRAINER.METALEARN.STEP 30 MODEL.INIT_WEIGHTS output/mscm_dist_net_retrain_digit5/svhn/model_tar/model.pth.tar-50'
CUDA_VISIBLE_DEVICES=$GPU python tools/train.py --root ../../data --trainer $trainer \
	--source-domains mnist mnist_m usps syn --target-domains svhn  \
	--dataset-config-file ${data_conf} --config-file ${conf}  \
	--output-dir output/mscm_dist_net_meta_learn_retrain_digit5/svhn --resume output/adaptkernel_digit5/usps/nomodel  \
	$opt 2>&1|tee output/mscm_dist_net_meta_learn_retrain_digit5/fm_mscm_retrain_svhn_${i}.log &
((GPU=GPU+1))
opt='MODEL.BACKBONE.NAME cnn_digit5_m3sda_ms TRAINER.RETRAIN.RATIO 1. TRAINER.CUTMIX.PROB 1. TRAINER.RETRAIN.EPOCH 20 OPTIM.MAX_EPOCH 20 TRAINER.METALEARN.TYPE sgd TRAINER.METALEARN.LR 0.00005 TRAINER.METALEARN.STEP 30 MODEL.INIT_WEIGHTS output/mscm_dist_net_retrain_digit5/syn/model_tar/model.pth.tar-50'
CUDA_VISIBLE_DEVICES=$GPU python tools/train.py --root ../../data --trainer $trainer \
	--source-domains mnist mnist_m svhn usps --target-domains syn  \
	--dataset-config-file ${data_conf} --config-file ${conf}  \
	--output-dir output/mscm_dist_net_meta_learn_retrain_digit5/syn --resume output/adaptkernel_digit5/usps/nomodel  \
	$opt 2>&1|tee output/mscm_dist_net_meta_learn_retrain_digit5/fm_mscm_retrain_syn_${i}.log &
done
