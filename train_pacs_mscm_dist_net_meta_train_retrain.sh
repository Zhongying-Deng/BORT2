trainer=FixMatchMSCMDistNetMetaLearnRetrain
outpath=pacs_fm_mscm_dist_net_meta_learn_retrain
for((i=0;i<=4;i++));
do
GPU=0
opt='MODEL.BACKBONE.NAME resnet18_ms TRAINER.RETRAIN.RATIO 0.95 OPTIM.MAX_EPOCH 50 TRAINER.CUTMIX.PROB 1. TRAINER.RETRAIN.EPOCH 50 TRAINER.METALEARN.TYPE sgd TRAINER.METALEARN.LR 0.00005 TRAINER.METALEARN.STEP 30 MODEL.INIT_WEIGHTS output/pacs_fm_mscm_dist_net_retrain/photo_t2/model_tar/model.pth.tar-150'
CUDA_VISIBLE_DEVICES=${GPU} python tools/train.py --root ../../data/ --trainer ${trainer} \
 --source-domains art_painting cartoon sketch --target-domains photo \
 --dataset-config-file configs/datasets/da/pacs.yaml \
 --config-file configs/trainers/da/pacs_staged_lr.yaml \
 --output-dir output/${outpath}/photo \
 --resume output/sketch/nomodel \
 $opt  2>&1|tee output/${outpath}/fm_mscm_photo_${i}.log &
((GPU=GPU+1))
opt='MODEL.BACKBONE.NAME resnet18_ms TRAINER.RETRAIN.RATIO 0.95 OPTIM.MAX_EPOCH 50 TRAINER.CUTMIX.PROB 1. TRAINER.RETRAIN.EPOCH 50 TRAINER.METALEARN.TYPE sgd TRAINER.METALEARN.LR 0.00005 TRAINER.METALEARN.STEP 30 MODEL.INIT_WEIGHTS output/pacs_fm_mscm_dist_net_retrain/sketch_t2/model_tar/model.pth.tar-150'
CUDA_VISIBLE_DEVICES=${GPU} python tools/train.py --root ../../data/ --trainer ${trainer} \
 --source-domains art_painting cartoon photo --target-domains sketch \
 --dataset-config-file configs/datasets/da/pacs.yaml \
 --config-file configs/trainers/da/pacs_staged_lr.yaml \
 --output-dir output/${outpath}/sketch \
 --resume output/sketch/nomodel \
 $opt 2>&1|tee output/${outpath}/fm_mscm_sketch_${i}.log &
((GPU=GPU+1))
opt='MODEL.BACKBONE.NAME resnet18_ms TRAINER.RETRAIN.RATIO 0.95 OPTIM.MAX_EPOCH 50 TRAINER.CUTMIX.PROB 1. TRAINER.RETRAIN.EPOCH 50 TRAINER.METALEARN.TYPE sgd TRAINER.METALEARN.LR 0.00005 TRAINER.METALEARN.STEP 30 MODEL.INIT_WEIGHTS output/pacs_fm_mscm_dist_net_retrain/art_painting_t2/model_tar/model.pth.tar-150'
CUDA_VISIBLE_DEVICES=${GPU} python tools/train.py --root ../../data/ --trainer ${trainer} \
 --source-domains photo cartoon sketch --target-domains art_painting \
 --dataset-config-file configs/datasets/da/pacs.yaml \
 --config-file configs/trainers/da/pacs_staged_lr.yaml \
 --output-dir output/${outpath}/art_painting \
 --resume output/sketch/nomodel \
 $opt 2>&1|tee output/${outpath}/fm_mscm_art_${i}.log &
((GPU=GPU+1))
opt='MODEL.BACKBONE.NAME resnet18_ms TRAINER.RETRAIN.RATIO 0.95 OPTIM.MAX_EPOCH 50 TRAINER.CUTMIX.PROB 1. TRAINER.RETRAIN.EPOCH 50 TRAINER.METALEARN.TYPE sgd TRAINER.METALEARN.LR 0.00005 TRAINER.METALEARN.STEP 30 MODEL.INIT_WEIGHTS output/pacs_fm_mscm_dist_net_retrain/cartoon_t2/model_tar/model.pth.tar-150'
CUDA_VISIBLE_DEVICES=${GPU} python tools/train.py --root ../../data/ --trainer ${trainer} \
 --source-domains art_painting photo sketch --target-domains cartoon \
 --dataset-config-file configs/datasets/da/pacs.yaml \
 --config-file configs/trainers/da/pacs_staged_lr.yaml \
 --output-dir output/${outpath}/cartoon \
 --resume output/sketch/nomodel \
 $opt 2>&1|tee output/${outpath}/fm_mscm_cartoon_${i}.log &
done
