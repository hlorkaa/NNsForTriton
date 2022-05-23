import os
from pathlib import Path
import cv2
import json
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import argparse
import math
import pandas as pd

import argparse
from functools import partial
import sys
from attrdict import AttrDict

#import tritonclient.grpc as grpcclient
#import tritonclient.grpc.model_config_pb2 as mc
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
from tritonclient.utils import triton_to_np_dtype

from tqdm import trange
from torch.utils.data import Dataset, SubsetRandomSampler, IterableDataset
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import time
from sklearn import metrics

if sys.version_info >= (3, 0):
    import queue
else:
    import Queue as queue





FLAGS = None


def parse_model_grpc(model_metadata, model_config):
    """
    Check the configuration of a model to make sure it meets the
    requirements for an image classification network (as expected by
    this client)
    """
    if len(model_metadata.inputs) != 1:
        raise Exception("expecting 1 input, got {}".format(
            len(model_metadata.inputs)))

    if len(model_config.input) != 1:
        raise Exception(
            "expecting 1 input in model configuration, got {}".format(
                len(model_config.input)))

    input_metadata = model_metadata.inputs[0]
    output_metadata = model_metadata.outputs

    return (input_metadata.name, output_metadata, model_config.max_batch_size)


def parse_model_http(model_metadata, model_config):
    """
    Check the configuration of a model to make sure it meets the
    requirements for an image classification network (as expected by
    this client)
    """
    if len(model_metadata['inputs']) != 1:
        raise Exception("expecting 1 input, got {}".format(
            len(model_metadata['inputs'])))

    if len(model_config['input']) != 1:
        raise Exception(
            "expecting 1 input in model configuration, got {}".format(
                len(model_config['input'])))

    input_metadata = model_metadata['inputs'][0]
    output_metadata = model_metadata['outputs']

    return (input_metadata['name'], output_metadata,
            model_config['max_batch_size'])


def postprocess(results, output_names, filenames, batch_size):
    """
    Post-process results to show classifications.
    """
    output_dict = {}
    for output_name in output_names:
        output_dict[output_name] = results.as_numpy(output_name)
        if len(output_dict[output_name]) != batch_size:
            raise Exception("expected {} results for output {}, got {}".format(
                batch_size, output_name, len(output_dict[output_name])))

    for n, f in enumerate(filenames):
        print('\n"{}":'.format(f))
        for output_name in output_names:
            print('  [{}]:'.format(output_name))
            for result in output_dict[output_name][n]:
                if output_dict[output_name][n].dtype.type == np.object_:
                    cls = "".join(chr(x) for x in result).split(':')
                else:
                    cls = result.split(':')
                print("    {} ({}) = {}".format(cls[0], cls[1], cls[2]))


# def recognize_image(image, model_name):
#     url = 'localhost:8000'
#     classes = 1
#     protocol = 'http'
#
#     try:
#          triton_client = httpclient.InferenceServerClient(url=url, verbose=False)
#     except Exception as e:
#         print("client creation failed: " + str(e))
#         sys.exit(1)
#
#     try:
#         model_metadata = triton_client.get_model_metadata(model_name=model_name)
#     except InferenceServerException as e:
#         print("failed to retrieve the metadata: " + str(e))
#         sys.exit(1)
#
#     try:
#         model_config = triton_client.get_model_config(model_name=model_name)
#     except InferenceServerException as e:
#         print("failed to retrieve the config: " + str(e))
#         sys.exit(1)
#
#     input_name, output_metadata, batch_size = parse_model_http(
#         model_metadata, model_config)

    # batch_size = 1





































class IteratorFromClip(object):
    """
    This class creates an iterator from a part of the video.
    On iteration, returns dict:
                     {'image':img, 'filename':frame_timestamp}
    """
    def __init__(self, video_path, start_frame, num_of_iterations, transform):
        """
        Init iterator

        Parametres
        -----
        video_path            - The path to the video
        start_frame           - The index of the frame at which part of
                                the video begins
        num_of_iterations     - The number of iterations for the video clip
        transform             - torchvision.transforms for frames

        """
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        self.num_of_iterations = num_of_iterations
        self.transform = transform

    def __iter__(self):
        """
        Returns iterator
        """
        return self

    def __next__(self):
        """
        While num_of_iterations>0 returns:
                                {'image':img, 'filename':frame_timestamp}

        """
        if self.num_of_iterations>0:
            frame_num = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            frame_timestamp = humanize_time(1000*frame_num / self.fps )
            ret, frame = self.cap.read()
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            if self.transform:
                frame = self.transform(frame)
            self.num_of_iterations-=1
            return {'image': frame, 'filename': frame_timestamp}
        else:
            self.cap.release()
            raise StopIteration()


class VideoDataset(IterableDataset):
    """
    This class is an IterableDataset of video.
    On iteration, returns dict:
                     {'image':img, 'filename':frame_timestamp}
    """
    def __init__(self, video_path, transform=None):
        """
        Init dataset

        Parametres
        -----
        video_path            - The path to the video
        transform             - torchvision.transforms for frames

        """
        super(VideoDataset).__init__()
        cap = cv2.VideoCapture(video_path)
        self.video_path = video_path
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        self.transform = transform


    def __iter__(self):
        """
        Return IteratorFromClip.
        if num_workers>0: Splits video into clips and returns iterators

        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            start_frame = 0
            num_of_iterations = self.frame_count
        else:
            num_of_iterations = int(math.ceil(self.frame_count / worker_info.num_workers))
            worker_id = worker_info.id
            start_frame = worker_id * num_of_iterations
            num_of_iterations = min(self.frame_count - start_frame, num_of_iterations)
        return IteratorFromClip(video_path=self.video_path,
                                  start_frame=start_frame,
                                  num_of_iterations=num_of_iterations,
                                  transform=self.transform )


def humanize_time(ms):
    """
    Convert time in ms to 'hh:mm:ss:ms'

    Parametres
    -----
    ms       -time in ms

    Returns
    -----
    'hh:mm:ss:ms'
    """
    secs, ms = divmod(ms, 1000)
    mins, secs = divmod(secs, 60)
    hours, mins = divmod(mins, 60)
    return '%02d:%02d:%02d:%03d' % (hours, mins, secs, ms)


def make_swd_by_video(video_path, scene_model_parameters, weather_model_parameters, batch_size, num_workers,
                      scene_by_index, weather_by_index, window_size, out_path):
    """
    Predicts scene and weather of each frame from the video.
    Smooth all predictions with sliding window.

    Parametres
    -----
    video_path              - The path to the video
    batch_size              - How many samples per batch to load.
    num_workers             - How many subprocesses to use for data loading.
    device                  - Device for computation. 'cuda' or 'cpu'
    model_path              - Path to .pt or .pth file with model weights.
    scene_by_index          - Dict to map scene label and index (e.g. {0:"City",...})
    weather_by_index        - Dict to map weather label and index (e.g. {0:"Clear",...})
    window_size             - Size of sliding window
    out_path                - Path to output folder with csv for each video
                              with predictions for each frame


    Output
    -----
    csv with predictions for each frame
    """
    video_name = os.path.split(video_path)[1]
    data = VideoDataset(video_path, transform=transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])]))
    loader = torch.utils.data.DataLoader(data,batch_size=batch_size, num_workers=num_workers)
    [prediction_scene, prediction_weather], timestamps = test_model(scene_model_parameters, weather_model_parameters, loader, return_gr_tr=False)
    timestamps, prediction_scene, prediction_weather = zip(*sorted(zip(timestamps, prediction_scene, prediction_weather)))
    smooth_scene_prediction, smooth_weather_prediction = [], []
    for i in range(len(timestamps)): # set i-th prediction as most frequent in sliding window
        right_border = i+int(window_size/2)+1
        left_border = max(0,right_border-window_size)
        scene_window = prediction_scene[left_border:right_border]
        weather_window = prediction_weather[left_border:right_border]
        scene = scene_by_index[max(set(scene_window), key = scene_window.count)]
        weather = weather_by_index[max(set(weather_window), key = weather_window.count)]
        smooth_scene_prediction.append(scene)
        smooth_weather_prediction.append(weather)
    df = pd.DataFrame({'timestamp': timestamps,
                       'scene': smooth_scene_prediction,
                       'weather': smooth_weather_prediction})
    out_filename = f'{video_name}.csv'
    df.to_csv(os.path.join(out_path, out_filename))
    print('Complete')


def get_prediction(input_name, output_metadata, model_name, image_data):
    inputs = []
    inputs.append(
        httpclient.InferInput(input_name, image_data.shape,
                              "BYTES"))
    inputs[0].set_data_from_numpy(image_data, binary_data=True)

    outputs = []
    output_names = [
        output['name']
        for output in output_metadata
    ]
    for output_name in output_names:
        outputs.append(
            httpclient.InferRequestedOutput(output_name,
                                            binary_data=True,
                                            class_count=1))

    result = triton_client.infer(model_name, inputs, outputs=outputs)
    print(result)
    return 0 #result


def test_model(scene_model_parameters, weather_model_parameters,  loader, return_gr_tr=False):
    """
    Tests the model. Can work with a model with multiple labels.

    Parametres
    -----
    model        - nn.model that returns 1 or more labels
    lodar        - DataLoader that returns a dictionary
                   {'image':, 'label':,'filename':} (label can be list or str)
    return_gr_tr - if True function return prediction and ground_truh
                   if False function return only prediction
    device       - by default torch.device('cuda:0')
    Returns
    -----
    if labels=True:  [ground_truh, prediction, filenames]
    if labels=True:  [prediction, filenames]
    """
    # model.to(device)
    # model.eval()
    pred = []
    filenames = list()
    with torch.no_grad():
        for loader_batch in loader:
            filenames.extend(loader_batch['filename'])
            image = loader_batch['image']
            image = np.asarray(image)
            image_data = np.stack(image, axis=0)

            predict_scene = get_prediction(scene_model_parameters['input'], scene_model_parameters['output'],
                                           scene_model_parameters['name'], image_data)
            predict_weather = get_prediction(weather_model_parameters['input'], weather_model_parameters['output'],
                                             weather_model_parameters['name'], image_data)
            predict = [predict_scene, predict_weather]

            # predict = model(image)
            if isinstance(predict, list): #If the model predicts more than one label
                indices = [(lambda label: torch.max(label, 1)[1].cpu().numpy())(label)
                            for label in predict]
                pred.extend(np.array(indices, dtype='int32').T)
                if return_gr_tr: #If we want to return the ground truth
                    to_np_labels = [(lambda label: label.numpy())(label)
                                    for label in loader_batch['label']]
                    #gr_tr.extend(np.array(to_np_labels).T)
            else:
                indices = torch.max(predict, 1)[1]
                pred.extend(indices.cpu().tolist())
            #if return_gr_tr:
                #gr_tr.extend(loader_batch['label'])
    #if return_gr_tr:
        # return [np.array(gr_tr).T, np.array(pred).T, np.array(filenames).T]
    #else:
    return [np.array(pred).T, np.array(filenames).T]


def get_model_parameters(model_name):
    try:
        model_metadata = triton_client.get_model_metadata(model_name=model_name)
    except InferenceServerException as e:
        print("failed to retrieve the metadata: " + str(e))
        sys.exit(1)

    try:
        model_config = triton_client.get_model_config(model_name=model_name)
    except InferenceServerException as e:
        print("failed to retrieve the config: " + str(e))
        sys.exit(1)

    input_name, output_metadata, batch_size = parse_model_http(model_metadata, model_config)

    return input_name, output_metadata, batch_size

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in_path', help='Path to the video or \
                                                 path to the folder with videos')
    parser.add_argument('-o', '--out_path', help="Path to output folder with csv for \
                                                  each video with predictions \
                                                  for each frame")
    parser.add_argument('-m', '--model_folder', help='Path to folder with .pt(h) file with \
                                           model weights and json files that map class name to index')
    parser.add_argument('-c', '--config_path', default= "SWD_config.json", help='The path to configuration file.\
                                                                             For more details, see README.md')
    args = parser.parse_args()
    in_path = args.in_path
    model_folder = args.model_folder
    out_path = args.out_path
    # config_path = args.config_path
    # with open(config_path, 'r') as config_json:
    #     config_dict = json.load(config_json)
    with open(os.path.join(model_folder, 'weather_dict.json'),'r') as weather_dict_file:
        weather_dict = json.load(weather_dict_file)
        weather_by_index = {int(i):key for i,key in weather_dict[0].items()}
    with open(os.path.join(model_folder, 'scene_dict.json'),'r') as scene_dict_file:
        scene_dict = json.load(scene_dict_file)
        scene_by_index = {int(i):key for i,key in scene_dict[0].items()}

    scene_model_name = 'scene_model'
    weather_model_name = 'weather_model'
    url = 'localhost:8000'
    classes = 1
    protocol = 'http'
    num_workers = 4 #config_dict['num_workers']
    window_size = 240 #config_dict['window_size']

    try:
        triton_client = httpclient.InferenceServerClient(url=url, verbose=False)
    except Exception as e:
        print("client creation failed: " + str(e))
        sys.exit(1)

    scene_input_name, scene_output_metadata, scene_batch_size = get_model_parameters(scene_model_name)
    weather_input_name, weather_output_metadata, weather_batch_size = get_model_parameters(weather_model_name)
    scene_model_parameters = {'name': scene_model_name, 'input': scene_input_name, 'output': scene_output_metadata}
    weather_model_parameters = {'name': weather_model_name, 'input': weather_input_name, 'output': weather_output_metadata}

    if scene_batch_size != weather_batch_size:
        print('models batch size should be same!')
        sys.exit(1)

    batch_size = scene_batch_size

    Path(out_path).mkdir(parents=True, exist_ok=True)
    # batch_size = config_dict['batch_size']

    # model_name = config_dict['model_name']
    # model = torch.load(os.path.join(model_folder, model_name))
    # if torch.cuda.is_available() and 'cuda' in config_dict['device']:
    #     device = torch.device(config_dict['device'])
    # else:
    #     device = torch.device('cpu')
    if os.path.isdir(in_path):
        video_names = os.listdir(in_path)
    else:
        in_path, video_name = os.path.split(in_path)
        video_names = [video_name]

    for video_name in video_names:
        print(f'Processing {video_name}...', end='\t')
        make_swd_by_video(video_path= os.path.join(in_path, video_name),
                          scene_model_parameters=scene_model_parameters,
                          weather_model_parameters=weather_model_parameters,
                          batch_size= batch_size,
                          num_workers= num_workers,
                          scene_by_index= scene_by_index,
                          weather_by_index= weather_by_index,
                          window_size= window_size,
                          out_path= out_path)
