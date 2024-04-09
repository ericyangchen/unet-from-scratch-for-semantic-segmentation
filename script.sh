# visualize training process
tensorboard --logdir="./outputs" --port 9000


subdir="20240409-1808"

# train
## UNet
### best: normal
### 20240409-1808 (w/o aug):  | 0.9171 | 0.9203 |
python train.py --seed 2024 --epochs 100 --batch_size 46 --lr 1e-3 --weight_decay 1e-4 --model UNet --save_dir ./outputs/UNet

## ResNet34_UNet 
### best: 20240409-1714 (w/o aug):  | 0.9015 | 0.9058 |
python train.py --seed 2024 --epochs 100 --batch_size 200 --lr 1e-3 --weight_decay 1e-4 --model ResNet34_UNet --save_dir ./outputs/ResNet34_UNet


# evaluate
python evaluate.py --model UNet --model_path ./outputs/UNet/${subdir}/UNet_best_model.pth
python evaluate.py --model ResNet34_UNet --model_path ./outputs/ResNet34_UNet/${subdir}/ResNet34_UNet_best_model.pth

# test (inference)
python inference.py --model UNet --model_path ./outputs/UNet/${subdir}/UNet_best_model.pth
python inference.py --model ResNet34_UNet --model_path ./outputs/ResNet34_UNet/${subdir}/ResNet34_UNet_best_model.pth

# compare best models
python compare_models.py --model_paths ./outputs/UNet/best ./outputs/ResNet34_UNet/best --output_dir ./outputs/comparisons/best


# In report, For TAs:
# train
python train.py --seed 2024 --epochs 100 --batch_size 128 --lr 1e-2 --weight_decay 1e-3 --model UNet --save_dir ./outputs/UNet
python train.py --seed 2024 --epochs 100 --batch_size 128 --lr 1e-2 --weight_decay 1e-3 --model ResNet34_UNet --save_dir ./outputs/ResNet34_UNet
# inference
python inference.py --model UNet --model_path ../saved_models/UNet_best_model.pth
python inference.py --model ResNet34_UNet --model_path ../saved_models/ResNet34_UNet_best_model.pth
