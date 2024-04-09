import os
import pytz
from tqdm import tqdm
import argparse
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

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


def train(args, model, train_dataset, valid_dataset):
    # init tensorboard
    writer = SummaryWriter(log_dir=f"{args.save_dir}/logs")

    # prepare data loader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)

    # prepare optimizer, scheduler and loss function
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = torch.nn.BCEWithLogitsLoss()

    # logs
    epoch_loss_list, epoch_accuracy_list = [], []
    valid_epoch_loss_list, valid_epoch_accuracy_list = [], []

    # save model
    best_valid_accuracy = 0.0
    best_model_checkpoint_path = os.path.join(
        args.save_dir, f"{model.__name__}_best_model.pth"
    )

    # training
    for epoch in tqdm(range(1, args.epochs + 1)):
        # train mode
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0

        for batch_data in train_loader:
            inputs, labels = batch_data["image"], batch_data["mask"]
            inputs, labels = inputs.to(args.device), labels.to(args.device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            preds = convert_output_to_binary_mask(outputs)
            dice_scores = dice_score(preds, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_accuracy += dice_scores.sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = running_accuracy / len(train_loader.dataset)

        epoch_loss_list.append(epoch_loss)
        epoch_accuracy_list.append(epoch_accuracy)

        writer.add_scalar("Train/Loss", epoch_loss, epoch)
        writer.add_scalar("Train/Accuracy", epoch_accuracy, epoch)

        # eval mode
        model.eval()
        running_valid_loss = 0.0
        running_valid_accuracy = 0.0

        tensorboard_visualization = None
        with torch.no_grad():
            for batch_data in valid_loader:
                inputs, labels = batch_data["image"], batch_data["mask"]
                inputs, labels = inputs.to(args.device), labels.to(args.device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                preds = convert_output_to_binary_mask(outputs)
                dice_scores = dice_score(preds, labels)

                running_valid_loss += loss.item() * inputs.size(0)
                running_valid_accuracy += dice_scores.sum().item()

                # randomly select 10 samples for visualization in tensorboard
                if tensorboard_visualization is None:
                    num_samples = 10 if len(inputs) >= 10 else len(inputs)

                    visualized_inputs = inputs[:num_samples]
                    visualized_labels = torch.cat(
                        [labels[:num_samples] * 255] * 3, dim=1
                    )
                    visualized_preds = torch.cat([preds[:num_samples] * 255] * 3, dim=1)

                    tensorboard_visualization = torch.cat(
                        [visualized_inputs, visualized_labels, visualized_preds], dim=3
                    )

        if tensorboard_visualization is not None:
            writer.add_images(
                "Validation/Visualization",
                tensorboard_visualization,
                epoch,
                dataformats="NCHW",
            )

        valid_epoch_loss = running_valid_loss / len(valid_loader.dataset)
        valid_epoch_accuracy = running_valid_accuracy / len(valid_loader.dataset)

        valid_epoch_loss_list.append(valid_epoch_loss)
        valid_epoch_accuracy_list.append(valid_epoch_accuracy)

        writer.add_scalar("Validation/Loss", valid_epoch_loss, epoch)
        writer.add_scalar("Validation/Accuracy", valid_epoch_accuracy, epoch)

        # save best model
        if valid_epoch_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_epoch_accuracy
            torch.save(model.state_dict(), best_model_checkpoint_path)

        # update learning rate
        scheduler.step()

        tqdm.write(
            f"Epoch {epoch}/{args.epochs}, loss: {epoch_loss:.4f}, acc: {epoch_accuracy:.4f}, val. loss: {valid_epoch_loss:.4f}, val. acc: {valid_epoch_accuracy:.4f}"
        )

    # Close TensorBoard writer
    writer.close()

    # Load the best model
    model.load_state_dict(torch.load(best_model_checkpoint_path))
    print(
        f"Loaded {model.__name__} model with best valid accuracy: {best_valid_accuracy:.4f}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Oxford-IIIT Pet Dataset Binary Segmentation with UNet / ResNet34+UNet"
    )

    parser.add_argument(
        "--data_path", type=str, default="../dataset", help="path to dataset"
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--model", type=str, default="UNet", choices=["UNet", "ResNet34_UNet"]
    )
    parser.add_argument("--save_dir", type=str, default=None, help="Save directory")
    parser.add_argument(
        "--mask_threshold", type=float, default=0.5, help="Mask threshold"
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # load args
    args = parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # set random seed
    if args.seed is not None:
        set_random_seed(args.seed)

    # create output directory
    if args.save_dir is not None:
        tz = pytz.timezone("Asia/Taipei")
        now = datetime.now(tz).strftime("%Y%m%d-%H%M")
        args.save_dir = f"{args.save_dir}/{now}"

        # create output directory
        os.makedirs(args.save_dir, exist_ok=True)

        # create logs directory
        os.makedirs(f"{args.save_dir}/logs", exist_ok=True)

    # load dataset
    train_dataset = OxfordPetDataset(root=args.data_path, mode="train")
    valid_dataset = OxfordPetDataset(root=args.data_path, mode="valid")

    # load model
    model = get_model(args)
    model = model.to(args.device)

    print(f"Start training {model.__name__}:")
    print(f"Output directory: {args.save_dir}\n")
    print(model)

    # train
    train(args, model, train_dataset, valid_dataset)
