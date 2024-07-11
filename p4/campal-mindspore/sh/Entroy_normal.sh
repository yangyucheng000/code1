#function run
run() {
    number=$1
    shift
    for i in `seq $number`; do
      python $@
    done
}

CUDA_VISIBLE_DEVICES=1 run 1 /home/jovyan/CAMPAL-mindspore/main.py \
    --model resnet18_cifar \
    --dataset svhn \
    --strategy EntropySampling \
    --num-init-labels 100 \
    --n-cycle 20 \
    --num-query 100 \
    --subset 10000 \
    --n-epoch 30 \
    --batch-size 50 \
    --lr 0.1 \
    --momentum 0.9 \
    --weight-decay 0.0005 \
    --milestones 120 180 240 \
    --aug-lab-on \
    --aug-ratio-lab 1 \
    --mix-ratio-lab 1 \
    --aug-lab-training-mode StrengthGuidedAugment \
    --aug-lab-strength-mode all \
    --aug-ulb-on \
    --aug-ratio-ulb 1 \
    --mix-ratio-ulb 1 \
    --aug-ulb-evaluation-mode StrengthGuidedAugment \
    --aug-ulb-strength-mode all \
    --aug-metric-ulb normal
