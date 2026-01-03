trainer=FixMatchCutMix
opt='MODEL.BACKBONE.NAME resnet18_ms TRAINER.CUTMIX.PROB 1. TRAINER.CUTMIX.BETA 1. TRAIN.CHECKPOINT_FREQ 10' 
outpath=pacs_fm_cutmix
for((i=0;i<=1;i++));
do
CUDA_VISIBLE_DEVICES=${GPU} python tools/train.py --root ../../data/ --trainer ${trainer} \
 --source-domains art_painting cartoon sketch --target-domains photo \
 --dataset-config-file configs/datasets/da/pacs.yaml \
 --config-file configs/trainers/da/pacs_staged_lr.yaml \
 --output-dir output/${outpath}/photo \
 --resume output/adaptkernel_pacs/sketch/nomodel \
 $opt  2>&1|tee output/${outpath}/fm_cutmix_photo_${i}.log &
((GPU=GPU+1))
CUDA_VISIBLE_DEVICES=${GPU} python tools/train.py --root ../../data/ --trainer ${trainer} \
 --source-domains art_painting cartoon photo --target-domains sketch \
 --dataset-config-file configs/datasets/da/pacs.yaml \
 --config-file configs/trainers/da/pacs_staged_lr.yaml \
 --output-dir output/${outpath}/sketch \
 --resume output/adaptkernel_pacs/sketch/nomodel \
 $opt  2>&1|tee output/${outpath}/fm_cutmix_sketch_${i}_FixMatch-CM_visualize.log &
((GPU=GPU+1))
CUDA_VISIBLE_DEVICES=${GPU} python tools/train.py --root ../../data/ --trainer ${trainer} \
 --source-domains photo cartoon sketch --target-domains art_painting \
 --dataset-config-file configs/datasets/da/pacs.yaml \
 --config-file configs/trainers/da/pacs_staged_lr.yaml \
 --output-dir output/${outpath}/art_painting \
 --resume output/adaptkernel_pacs/sketch/nomodel \
 $opt  2>&1|tee output/${outpath}/fm_cutmix_art_${i}.log &
((GPU=GPU+1))
CUDA_VISIBLE_DEVICES=${GPU} python tools/train.py --root ../../data/ --trainer ${trainer} \
 --source-domains art_painting photo sketch --target-domains cartoon \
 --dataset-config-file configs/datasets/da/pacs.yaml \
 --config-file configs/trainers/da/pacs_staged_lr.yaml \
 --output-dir output/${outpath}/cartoon \
 --resume output/adaptkernel_pacs/sketch/nomodel \
 $opt  2>&1|tee output/${outpath}/fm_cutmix_cartoon_${i}.log &
done
