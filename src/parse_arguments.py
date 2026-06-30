import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()

    # === Base Arguments ===
    parser.add_argument("--exp_name", 
                        type=str, 
                        help="Experiment to evaluate")
    parser.add_argument("--device", 
                        type=int, 
                        default=0, 
                        help="GPU ID to use.")
    parser.add_argument("--preload", 
                        nargs='+', 
                        type=str, 
                        default=['captions', 'mods'], 
                        help="List of properties to preload (computed once before).")

    # === Base Model Choices ===
    parser.add_argument("--clip", 
                        type=str, 
                        default='ViT-B-32', 
                        choices=['ViT-Base/32', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 
                                 'ViT-bigG-14', 'ViT-B-32', 'ViT-B-16', 'ViT-L-14', 'ViT-H-14', 'ViT-g-14'],
                        help="Which CLIP text-to-image retrieval model to use.")

    # === Dataset Arguments ===
    parser.add_argument("--dataset", 
                        default="webvidcovr", 
                        type=str, 
                        required=False, 
                        help="Dataset to use")
    parser.add_argument("--split", 
                        type=str, 
                        default='test', 
                        choices=['val', 'test'],
                        help='Dataset split to evaluate on. Some datasets require special testing protocols like cirr/circo.')
    parser.add_argument("--dataset_path", 
                        default="../datasets/WebVidCOVR", 
                        type=str, 
                        required=False,
                        help="Path to the dataset")

    return parser.parse_args()
