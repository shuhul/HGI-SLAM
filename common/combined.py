import bowhandler as bowh
import bagofwords as bow
import handler
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Run HGI on a image sequence')
    parser.add_argument('path_to_sequence', type=str)
    args = parser.parse_args()

    sequence_folder = args.path_to_sequence

    handler.showTrajectory(start=0, stop=-1, showGT=False, showB=True)

    # handler.readFolder(sequence_folder, 'saved')
    # print(handler.getFrameNumber('1305031098.6659'))
    # print(handler.timestamps)
    # loop_closure_connections = bow.getLCC()
    # handler.saveLoopClosures(loop_closure_connections)
    # print(loop_closure_connections)
    
    # bowh.combined(sequence_folder, num_frames=300, detecting=False, sup_weight=1, sal_weight=1, sim_threshold=1)