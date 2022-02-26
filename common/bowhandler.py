import common.handler as handler
import common.bagofwords as bow



def run(sequence_folder, featureExtractor, max_frame=750, training=True, num_clusters=3, num_neighbors=3, detecting=True, max_distance=1):

    print('\n-------Generating Descriptors--------\n')

    handler.readFolder(sequence_folder) 

    descriptor_list = handler.readDescriptors()

    print(f'Starting number of descriptors: {len(descriptor_list)}')

    skip = 4

    batch_size = 5

    num_frames = int(max_frame/skip)

    if handler.readCurrentIndex() >= max_frame:
        print(f'Skipping {handler.readCurrentIndex()} already processed frames')
    else:
        for i in range(int(num_frames/batch_size)):
            filenames, new_frames, last = handler.getNewFrames(last=(i+1)*batch_size*skip, skip=skip)
            descriptor_list = handler.readDescriptors(max=i*batch_size) + featureExtractor(new_frames)
            if len(descriptor_list) >= (batch_size) and len(new_frames) != 0:
                print(f'Saving new descriptors for batch: {i+1} of {int(num_frames/batch_size)}')
                handler.saveDescriptors(descriptor_list)
                handler.saveCurrentIndex(last)
        
    
    descriptor_list = handler.readDescriptors()

    print(f'Ending number of descriptors: {len(descriptor_list)}')

    print('\n-------Computing Bag Of Words--------\n')

    if training:
        bow.trainBoW(descriptor_list, n_clusters=num_clusters, n_neighbors=num_neighbors)
    else:
        print('Skipping already computed BoW model')

    # print('\n-------Detecting Loop Closures--------\n')
    
    # if detecting:
    #     bow.detectLoopClosures(descriptor_list, max_distance, max=num_frames)
    # else:
    #     print('Skipping already detected loop closures')

    # loop_closure_indices = bow.getLoopClosures()
    
    # loop_closure_connections = bow.getLCC()

    # print(f'\n-------Detected {len(loop_closure_indices)} loop closures--------\n')

    # if len(loop_closure_indices) != 0:

    #     print(f'Detected loop closures between indices {loop_closure_connections}\n')

    #     # handler.showLoopClosurePairs(loop_closure_connections)

    # else:
    #     print('No loop closures found')


def combined(sequence_folder, num_frames=750, detecting=True, sup_weight=1, sal_weight=1, sim_threshold=1):

    distance_threshold = 1/sim_threshold

    skip = 4

    handler.readFolder(sequence_folder) 

    print('\n-------Detecting Loop Closures--------\n')

    if detecting:
        bow.detectCombinedLC(sup_weight, sal_weight, distance_threshold, max_frames=num_frames, skip=skip)
    else:
        print('Skipping already detected loop closures')
    
    
    loop_closure_connections = bow.getLCC()
    
    print(f'\n-------Detected {len(loop_closure_connections)} loop closures--------\n')

    if len(loop_closure_connections) != 0:

        print(f'Detected loop closures between indices {loop_closure_connections}\n')

        if detecting:
            handler.showLoopClosurePairs(loop_closure_connections)

    else:
        print('No loop closures found')
    
