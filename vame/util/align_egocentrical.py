"""
Variational Animal Motion Embedding 0.1 Toolbox
© K. Luxem & J. Kürsch & P. Bauer, Department of Cellular Neuroscience
Leibniz Institute for Neurobiology, Magdeburg, Germany

https://github.com/LINCellularNeuroscience/VAME
Licensed under GNU General Public License v3.0
"""

import os
import cv2 as cv
import numpy as np
import pandas as pd
import tqdm
import sklearn
from scipy.signal import medfilt

from pathlib import Path
from vame.util.auxiliary import read_config  


# Returns cropped image using rect tuple
def crop_and_flip(rect, src, points, ref_index):
    # Read out rect structures and convert
    center, size, theta = rect
    center, size = tuple(map(int, center)), tuple(map(int, size))
    # Get rotation matrix
    M = cv.getRotationMatrix2D(center, theta, 1)
       
    # shift DLC points
    x_diff = center[0] - size[0]//2
    y_diff = center[1] - size[1]//2
    
    dlc_points_shifted = []
    
    for i in points:
        point = cv.transform(np.array([[[i[0], i[1]]]]), M)[0][0]

        point[0] -= x_diff
        point[1] -= y_diff
        
        dlc_points_shifted.append(point)
        
    # Perform rotation on src image
    dst = cv.warpAffine(src.astype('float32'), M, src.shape[:2])
    out = cv.getRectSubPix(dst, size, center)
    
    # check if flipped correctly, otherwise flip again
    if dlc_points_shifted[ref_index[1]][0] >= dlc_points_shifted[ref_index[0]][0]:
        rect = ((size[0]//2, size[0]//2), size, 180)
        center, size, theta = rect
        center, size = tuple(map(int, center)), tuple(map(int, size))
        # Get rotation matrix
        M = cv.getRotationMatrix2D(center, theta, 1)

        # shift DLC points
        x_diff = center[0] - size[0]//2
        y_diff = center[1] - size[1]//2
        
        points = dlc_points_shifted
        dlc_points_shifted = []
        
        for i in points:
            point = cv.transform(np.array([[[i[0], i[1]]]]), M)[0][0]
    
            point[0] -= x_diff
            point[1] -= y_diff
            
            dlc_points_shifted.append(point)
    
        # Perform rotation on src image
        dst = cv.warpAffine(out.astype('float32'), M, out.shape[:2])
        out = cv.getRectSubPix(dst, size, center)
        
    return out, dlc_points_shifted


# Helper function to return indexes of nans
def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0] 


# Interpolates all nan values of given array
def interpol(arr):
        
    y = np.transpose(arr)
     
    nans, x = nan_helper(y[0])
    y[0][nans] = np.interp(x(nans), x(~nans), y[0][~nans])
    nans, x = nan_helper(y[1])
    y[1][nans] = np.interp(x(nans), x(~nans), y[1][~nans])
    
    arr = np.transpose(y)
    
    return arr


def background(path_to_file, filename, video_format='.mp4', num_frames=1000):
    """
    Compute background image from fixed camera 
    """
    import scipy.ndimage
    capture = cv.VideoCapture(os.path.join(path_to_file, 'videos', filename+video_format))
    
    if not capture.isOpened():
        raise Exception("Unable to open video file: {0}".format(os.path.join(path_to_file, 'videos', filename+video_format)))
        
    frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    ret, frame = capture.read()
    
    height, width, _ = frame.shape    
    frames = np.zeros((height,width,num_frames))

    for i in tqdm.tqdm(range(num_frames), disable=not True, desc='Compute background image for video %s' %filename):
        rand = np.random.choice(frame_count, replace=False)
        capture.set(1, rand)
        ret, frame = capture.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frames[..., i] = gray
    
    print('Finishing up!')
    medFrame = np.median(frames, 2)
    background = scipy.ndimage.median_filter(medFrame, (5, 5))
    
    # np.save(path_to_file+'videos/'+'background/'+filename+'-background.npy',background)
    
    capture.release()
    return background


def align_mouse(path_to_file, filename, video_format, crop_size, pose_list,
                pose_ref_index, confidence, pose_flip_ref, bg, frame_count,
                use_video=True, interp_flag=True):
    """
    returns: list of cropped images (if video is used) and list of cropped DLC points

    parameters:
        path_to_file: directory
        filename: name of video file without format
        video_format: format of video file
        crop_size: tuple of x and y crop size
        dlc_list: list of arrays containg corresponding x and y DLC values
        dlc_ref_index: indices of 2 lists in dlc_list to align mouse along
        dlc_flip_ref: indices of 2 lists in dlc_list to flip mouse if flip was false
        bg: background image to subtract
        frame_count: number of frames to align
        use_video: boolean if video should be cropped or DLC points only
    """
    images = []
    points = []

    # interpolate if the flag is present
    if interp_flag:
        for i in pose_list:
            for j in i:
                if j[2] <= confidence:
                    j[0], j[1] = np.nan, np.nan

        for i in pose_list:
            i = interpol(i)
    
    if use_video:
        capture = cv.VideoCapture(os.path.join(path_to_file, 'videos', filename+video_format))

        if not capture.isOpened():
            raise Exception("Unable to open video file: {0}".format(os.path.join(path_to_file,
                                                                                 'videos', filename+video_format)))
          
    for idx in tqdm.tqdm(range(frame_count), disable=not True, desc='Align frames'):
        
        if use_video:
            # Read frame
            try:
                ret, frame = capture.read()
                frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                frame = frame - bg
                frame[frame <= 0] = 0
            except:
                print("Couldn't find a frame in capture.read(). #Frame: %d" % idx)
                continue
        else:
            frame = np.zeros((1, 1))
            
        # Read coordinates and add border
        pose_list_bordered = []
                
        for i in pose_list:
            pose_list_bordered.append((int(i[idx][0] + crop_size[0]), int(i[idx][1] + crop_size[1])))
        
        img = cv.copyMakeBorder(frame, crop_size[1], crop_size[1], crop_size[0], crop_size[0], cv.BORDER_CONSTANT, 0)
        
        punkte = []
        for i in pose_ref_index:
            # coord = []
            # coord.append(pose_list_bordered[i][0])
            # coord.append(pose_list_bordered[i][1])
            coord = [pose_list_bordered[i][0], pose_list_bordered[i][1]]
            punkte.append(coord)
        punkte = [punkte]
        punkte = np.asarray(punkte)
        
        # calculate minimal rectangle around snout and tail
        rect = cv.minAreaRect(punkte)
    
        # change size in rect tuple structure to be equal to crop_size
        lst = list(rect)
        lst[1] = crop_size
        rect = tuple(lst)
        
        center, size, theta = rect
        
        # crop image
        out, shifted_points = crop_and_flip(rect, img, pose_list_bordered, pose_flip_ref)
        
        if use_video:  # for memory optimization, just save images when video is used.
            images.append(out)
        points.append(shifted_points)
        
    if use_video:
        capture.release()
    
    time_series = np.zeros((len(pose_list) * 2, frame_count))
    for i in range(frame_count):
        idx = 0
        for j in range(len(pose_list)):
            time_series[idx:idx + 2, i] = points[i][j]
            idx += 2
        
    return images, points, time_series


# play aligned video
def play_aligned_video(a, n, frame_count):
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
              (0, 255, 255), (0, 0, 0), (255, 255, 255), (127, 0, 127), (0, 127, 127)]
    
    for i in range(frame_count):
        # Capture frame-by-frame
        ret, frame = True, a[i]
        if ret:

            # Display the resulting frame
            frame = cv.cvtColor(frame.astype('uint8') * 255, cv.COLOR_GRAY2BGR)
            im_color = cv.applyColorMap(frame, cv.COLORMAP_JET)

            for c, j in enumerate(n[i]):
                cv.circle(im_color, (j[0], j[1]), 5, colors[c], -1)

            cv.imshow('Frame', im_color)

            # Press Q on keyboard to  exit
            if cv.waitKey(25) & 0xFF == ord('q'):
                break
        
        # Break the loop
        else: 
            break
    cv.destroyAllWindows()


def alignment(path_to_file, filename, pose_ref_index, video_format,
              crop_size, confidence, use_video=False, check_video=False,
              column_list=None, dataframe=None):
    
    # read out data
    # process input differentially depending on the data type
    if dataframe is not None:
        # the path input is actually the data
        data = dataframe

        # set interpolation flag
        interp_flag = False
        # define the multiplier for the coordinates
        coord_multiplier = 1000
    # elif '_dlc.h5' in filename:
    #     data = pd.read_hdf(filename)
    #
    #     # set interpolation flag
    #     interp_flag = True
    #     # define the multiplier for the coordinates
    #     coord_multiplier = 1
    else:
        data = pd.read_csv(os.path.join(path_to_file, 'videos', 'pose_estimation', filename + '.csv'),
                           skiprows=2)
        # define the multiplier for the coordinates
        coord_multiplier = 1
        # set interpolation flag
        interp_flag = False

    # if column list is not none, select columns
    if column_list is not None:
        data = data.loc[:, column_list]
    # convert to numpy and multiply by 1000 to avoid rounding artifacts
    data_mat = pd.DataFrame.to_numpy(data)*coord_multiplier
    # get the coordinates for alignment from data table
    pose_list = []

    # select the factor depending on the interpolation
    if interp_flag:
        factor = 3
    else:
        factor = 2
    
    for i in range(int(data_mat.shape[1]/factor)):
        pose_list.append(data_mat[:, i*factor:(i+1)*factor])
        
    # list of reference coordinate indices for alignment
    # 0: snout, 1: forehand_left, 2: forehand_right,
    # 3: hindleft, 4: hindright, 5: tail
    
    pose_ref_index = pose_ref_index
    
    # list of 2 reference coordinate indices for avoiding flipping
    pose_flip_ref = pose_ref_index
        
    if use_video:
        # compute background
        bg = background(path_to_file, filename)
        capture = cv.VideoCapture(os.path.join(path_to_file, 'videos', filename + video_format))
        if not capture.isOpened():
            raise Exception("Unable to open video file: {0}".format(os.path.join(path_to_file,
                                                                                 'videos', filename + video_format)))
            
        frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
        capture.release()
    else:
        bg = 0
        frame_count = len(data)  # Change this to an arbitrary number if you first want to test the code

    frames, n, time_series = align_mouse(path_to_file, filename, video_format, crop_size, pose_list,
                                         pose_ref_index, confidence, pose_flip_ref, bg, frame_count,
                                         use_video=use_video, interp_flag=interp_flag)
    
    if check_video:
        play_aligned_video(frames, n, frame_count)
        
    return time_series/coord_multiplier, frames


def egocentric_alignment(config, pose_ref_index=[0, 5], crop_size=(300, 300), use_video=False,
                         video_format='.mp4', check_video=False, save_flag=True, filename=None,
                         column_list=None, dataframe=None, extra_columns=None):
    """ Happy aligning
    filename:   name root for the data
     """
    # config parameters
    # load the file if it's a path, otherwise it should be a dict
    if isinstance(config, str):
        config_file = Path(config).resolve()
        cfg = read_config(config_file)
    elif isinstance(config, dict):
        cfg = config
    else:
        raise TypeError("format of config not recognized")
    
    path_to_file = cfg['project_path']
    if filename is None:
        filename = cfg['video_sets']
    confidence = cfg['pose_confidence']
    # video_format = video_format
    # crop_size = crop_size

    egocentric_time_series = []
    frames = []
    
    # call function and save into your VAME data folder
    for idx, file in enumerate(filename):

        if dataframe is not None:
            current_data = dataframe[idx]
        else:
            current_data = None
        # print("Aligning data %s, Pose confidence value: %.2f" % (file, confidence))
        egocentric_time_series, frames = alignment(path_to_file, file, pose_ref_index, video_format, crop_size, 
                                                   confidence, use_video=use_video, check_video=check_video,
                                                   column_list=column_list, dataframe=current_data)
        if extra_columns is not None:
            # normalize the extra columns
            new_columns = current_data[extra_columns].to_numpy().T
            # smooth
            new_columns = medfilt(new_columns, 21)
            # new_columns = (new_columns - new_columns.min())/(new_columns.max() - new_columns.min())
            new_columns = sklearn.preprocessing.minmax_scale(new_columns)
            egocentric_time_series = np.concatenate([egocentric_time_series, new_columns], axis=0)

        if save_flag:
            np.save(os.path.join(path_to_file, 'data', file, file+'-PE-seq.npy'), egocentric_time_series)
#        np.save(os.path.join(path_to_file,'data/',file,"",file+'-PE-seq.npy', egocentric_time_series))

    return egocentric_time_series, frames
    # print("Your data is now in the right format and you can call vame.create_trainset()")

