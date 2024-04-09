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


def inference(args, model, test_dataset):
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # testing
    model.eval()

    test_accuracy = 0.0

    with torch.no_grad():
        for batch_data in test_loader:
            inputs, labels = batch_data["image"], batch_data["mask"]
            inputs, labels = inputs.to(args.device), labels.to(args.device)

            outputs = model(inputs)

            preds = convert_output_to_binary_mask(outputs)
            dice_scores = dice_score(preds, labels)

            test_accuracy += dice_scores.sum().item()

    test_accuracy = test_accuracy / len(test_loader.dataset)

    return test_accuracy


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
    test_dataset = OxfordPetDataset(root=args.data_path, mode="test")

    # load model
    model = get_model(args)
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(args.device)

    print(f"Testing model: {model.__name__}\n\tpath: {args.model_path}")

    # test
    test_accuracy = inference(args, model, test_dataset)
    print(f"Test accuracy (Dice Score): {test_accuracy:.4f}")
