import os
import pytz
from tqdm import tqdm
import argparse
from datetime import datetime
import torch
from torch.utils.data import DataLoader

from oxford_pet import OxfordPetDataset
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
from utils import set_random_seed, dice_score, convert_output_to_binary_mask


def get_model(args):
    if args.model == "UNet":
        model = UNet()
    elif args.model == "ResNet34_UNet":
        model = ResNet34_UNet()
    else:
        raise ValueError("Invalid model name")

    return model


def evaluate(args, model, valid_dataset):
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    # eval mode
    model.eval()

    valid_accuracy = 0.0

    with torch.no_grad():
        for batch_data in valid_loader:
            inputs, labels = batch_data["image"], batch_data["mask"]
            inputs, labels = inputs.to(args.device), labels.to(args.device)

            outputs = model(inputs)

            preds = convert_output_to_binary_mask(outputs)
            dice_scores = dice_score(preds, labels)

            valid_accuracy += dice_scores.sum().item()

    valid_accuracy = valid_accuracy / len(valid_loader.dataset)

    return valid_accuracy


def parse_args():
    parser = argparse.ArgumentParser(
        description="Oxford-IIIT Pet Dataset Binary Segmentation with UNet / ResNet34+UNet"
    )

    parser.add_argument(
        "--data_path", type=str, default="../dataset", help="path to dataset"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--model", type=str, default="UNet", choices=["UNet", "ResNet34_UNet"]
    )
    parser.add_argument("--model_path", type=str, default=None, help="Model path")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # load args
    args = parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # load dataset
    valid_dataset = OxfordPetDataset(root=args.data_path, mode="valid")

    # load model
    model = get_model(args)
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(args.device)

    print(f"Evaluating model: {model.__name__}\n\tpath: {args.model_path}")

    # evaluate (on valid data)
    valid_accuracy = evaluate(args, model, valid_dataset)
    print(f"Validation accuracy (Dice Score): {valid_accuracy:.4f}")
