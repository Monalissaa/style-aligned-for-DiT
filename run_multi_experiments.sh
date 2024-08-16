#!/bin/bash

# # 循环遍历 0 到 30
# for specific_share_layer in {0..39}; do
#     echo "Running with specific_share_layer=${specific_share_layer}"
#     python style_aligned_hunyuan.py --specific_share_layer ${specific_share_layer}
# done

# 循环遍历 0 到 39
for specific_share_layer in {15..39}; do
    echo "Running with specific_share_layer=${specific_share_layer}"
    python style_aligned_transfer_hunyuan.py --specific_share_layer ${specific_share_layer}
done

# for specific_share_layer in {0..39}; do
#     echo "Running with specific_share_layer=${specific_share_layer}"
#     python style_aligned_transfer_hunyuan_corgi.py --specific_share_layer ${specific_share_layer}
# done

# # 循环遍历 0 到 39
# for specific_share_layer in {0..39}; do
#     echo "Running with specific_share_layer=${specific_share_layer}"
#     python style_aligned_transfer_hunyuan.py --specific_share_layer ${specific_share_layer}
# done

# for specific_share_layer in {0..39}; do
#     echo "Running with specific_share_layer=${specific_share_layer}"
#     python style_aligned_transfer_hunyuan_corgi_adain_value.py --specific_share_layer ${specific_share_layer}
# done

python style_aligned_transfer_hunyuan_corgi_front20.py

