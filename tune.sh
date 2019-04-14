#!/usr/bin/env bash
device="1"
n_history="2"
d_mark="t"
d_mark_ptr="f"
mark_size="50"
for hidden_size in 128 256 400 512
    do
        echo "hidden_size = $hidden_size"
        CUDA_VISIBLE_DEVICES=$device python36 rc/main.py --trainset data_qp/coqa.train.json --devset data_qp/coqa.dev.json --n_history $n_history --dir qp_models_emb_tuning/hidden$hidden_size-mark$mark_size-adamax --embed_file wordvecs/glove.840B.300d.txt --predict_raw_text n --doc_mark_embed $d_mark --doc_mark_size $mark_size --doc_mark_in_pointer_computation $d_mark_ptr
    done

echo "tuning for $model finished!"

# n layers
# hidden size
# batch size
# history num
# sgd, lr, reduce
# momentum