import os
import argparse
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def extract_training_logs(log_dir):
    """
    Extract training logs from tensorboard data.

    Args:
        log_dir (str): The directory where tensorboard data is saved.

    Returns:
        dict: A dictionary containing training logs.
    """
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    scalars = ea.Tags()["scalars"]

    # load tensorboard data
    logs = {}
    for scalar in scalars:
        if scalar not in logs:
            logs[scalar] = []

        for event in ea.Scalars(scalar):
            logs[scalar].append((event.step, event.value))

    return logs


def get_values_from_events(events):
    return [event[1] for event in events]


def plot_model_accuracies(data, save_path=None):
    """
    Args:
        data (dict): dictionary containing model names as keys and their accuracies as values
            e.g, data = {
                            "ResNet50_train": [0.1, 0.2, 0.3, 0.4],
                            "ResNet50_valid": [0.1, 0.2, 0.3, 0.4],
                            "VGGNet19_train": [0.2, 0.3, 0.4, 0.5],
                            "VGGNet19_valid": [0.2, 0.3, 0.4, 0.5]
                        }
    """

    for model_name, model_accuracy in data.items():
        plt.plot(model_accuracy, label=model_name)

    plt.title("Accuracy Curve b/w Models")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    if save_path:
        plt.savefig(f"{save_path}/accuracy_curve.png")


def compare_models(args):
    model_paths = args.model_paths

    model_names = []
    identify_models_with_id = False

    data = []
    for model_path in model_paths:
        paths = model_path.strip().strip("/").split("/")
        model_name, model_id = paths[-2], paths[-1]

        # check for duplicated model names (ResNet50 or VGGNet19)
        if model_name in model_names:
            identify_models_with_id = True
        elif not identify_models_with_id:
            model_names.append(model_name)

        # transform logs
        logs = extract_training_logs(os.path.join(model_path, "logs"))
        logs["model_name"] = model_name
        logs["model_id"] = model_id
        logs["Train/Accuracy"] = get_values_from_events(logs["Train/Accuracy"])
        logs["Validation/Accuracy"] = get_values_from_events(
            logs["Validation/Accuracy"]
        )
        logs["Train/Loss"] = get_values_from_events(logs["Train/Loss"])
        logs["highest_train_accuracy"] = max(logs["Train/Accuracy"])

        data.append(logs)

    # plot accuracy
    plot_data = dict()
    for model in data:
        # set legends
        if identify_models_with_id:
            legend_train = f"{model['model_name']} ({model['model_id']}) - train"
            legend_valid = f"{model['model_name']} ({model['model_id']}) - valid"
        else:
            legend_train = f"{model['model_name']} - train"
            legend_valid = f"{model['model_name']} - valid"

        plot_data[legend_train] = model["Train/Accuracy"]
        plot_data[legend_valid] = model["Validation/Accuracy"]

    plot_model_accuracies(data=plot_data, save_path=args.output_dir)

    # print model info
    seperator = "-----------------------------------------"
    print(seperator)
    for model in data:
        if identify_models_with_id:
            print(f"Model: {model['model_name']} ({model['model_id']})")
        else:
            print(f"Model: {model['model_name']}")

        print(
            f"Train Accuracy: {model['highest_train_accuracy'] * 100:.2f}%, Train Loss: {model['Train/Loss'][-1]:.4f}"
        )
        print(seperator)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare models")

    parser.add_argument(
        "--data", type=str, default="../dataset", help="Path to dataset"
    )
    parser.add_argument(
        "--model_paths",
        type=str,
        nargs="+",
        help="Paths to model directories",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/comparisons",
        help="Path to comparison output",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    # create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # compare models
    compare_models(args)
