import common.handler as handler
import common.bagofwords as bow


def run(sequence_folder, featureExtractor):

    saved_folder = 'saved'

    for i in range(4):
        print('\n-------Generating Descriptors--------\n')

        handler.readFolder(sequence_folder, saved_folder)

        num_frames = 150 + (50*i) # max 750

        filenames, new_frames = handler.getNewFrames(last=num_frames)

        descriptor_list = handler.readDescriptors() + featureExtractor(new_frames)

        handler.saveDescriptors(descriptor_list)

    # print('\n-------Computing Bag Of Words--------\n')

    # training = False

    # if training:
    #     bow.trainBoW(descriptor_list, n_clusters=3, n_neighbors=3)
    # else:
    #     print('Skipping already computed BoW model')

    # print('\n-------Detecting Loop Closures--------\n')
    
    # detecting = True

    # min_distance = 1
    
    # if detecting:
    #     bow.detectLoopClosures(descriptor_list, min_distance)
    # else:
    #     print('Skipping already detected loop closures')

    # loop_closure_indices = bow.getLoopClosures()
    
    # loop_closure_connections = bow.getLCC()

    # print(f'\n-------Detected {len(loop_closure_indices)} loop closures--------\n')

    # if len(loop_closure_indices) != 0:

    #     print(f'Detected loop closures between indices {loop_closure_connections}\n')

    #     handler.showLoopClosurePairs(loop_closure_connections)

    # else:
    #     print('No loop closures found')