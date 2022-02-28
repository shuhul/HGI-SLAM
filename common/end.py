import bowhandler as bowh
import bagofwords as bow
import handler
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Run HGI on a image sequence')
    parser.add_argument('path_to_sequence', type=str)
    parser.add_argument('training', type=str)
    args = parser.parse_args()

    sequence_folder = args.path_to_sequence
    showLC = args.training == "y"
    train = True if args.training == "m" or showLC else False
    handler.readFolder(sequence_folder)
    handler.showTrajectory(showGT=True, showLC=showLC, create=train)