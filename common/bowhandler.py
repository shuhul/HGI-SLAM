import common.handler as handler
import common.bagofwords as bow


def run(sequence_folder, featureExtractor, num_frames=750, training=True, num_clusters=3, num_neighbors=3, detecting=True, max_distance=1):

    saved_folder = 'saved'

    print('\n-------Generating Descriptors--------\n')

    handler.readFolder(sequence_folder, saved_folder) 

    filenames, new_frames = handler.getNewFrames(last=num_frames)

    descriptor_list = handler.readDescriptors(max=num_frames) + featureExtractor(new_frames)

    if len(descriptor_list) > num_frames and len(new_frames) != 0:
        print('Saving new descriptors')
        handler.saveDescriptors(descriptor_list)
    
    print(f'Number of descriptors: {len(descriptor_list)}')

    print('\n-------Computing Bag Of Words--------\n')

    if training:
        bow.trainBoW(descriptor_list, n_clusters=num_clusters, n_neighbors=num_neighbors)
    else:
        print('Skipping already computed BoW model')

    print('\n-------Detecting Loop Closures--------\n')
    
    if detecting:
        bow.detectLoopClosures(descriptor_list, max_distance, max=num_frames)
    else:
        print('Skipping already detected loop closures')

    loop_closure_indices = bow.getLoopClosures()
    
    loop_closure_connections = bow.getLCC()

    print(f'\n-------Detected {len(loop_closure_indices)} loop closures--------\n')

    if len(loop_closure_indices) != 0:

        print(f'Detected loop closures between indices {loop_closure_connections}\n')

        handler.showLoopClosurePairs(loop_closure_connections)

    else:
        print('No loop closures found')


def combined(num_frames=750, detecting=True, sup_weight=1, sal_weight=1, sim_threshold=1):
    distance_threshold = 1/sim_threshold

    if detecting:
        bow.detectCombinedLC(sup_weight, sal_weight, distance_threshold, num_frames)
    else:
        print('Skipping already detected loop closures')
    
    loop_closure_indices = bow.getLoopClosures()
    
    loop_closure_connections = bow.getLCC()
    
    print(f'\n-------Detected {len(loop_closure_indices)} loop closures--------\n')

    if len(loop_closure_indices) != 0:

        print(f'Detected loop closures between indices {loop_closure_connections}\n')

        handler.showLoopClosurePairs(loop_closure_connections)

    else:
        print('No loop closures found')
    
