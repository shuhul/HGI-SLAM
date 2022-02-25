import bowhandler as bowh
import bagofwords as bow
import handler
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Run HGI on a image sequence')
    parser.add_argument('path_to_sequence', type=str)
    parser.add_argument('num_imgs', type=int)
    parser.add_argument('training', type=str)
    args = parser.parse_args()

    sequence_folder = args.path_to_sequence
    num = args.num_imgs
    train = True if args.training == "y" else False
    
    bowh.combined(sequence_folder, num_frames=num, detecting=train, sup_weight=1, sal_weight=1, sim_threshold=1)