import numpy as np
import cv2
import torch
from torchvision import transforms
from collections import Counter

classes = {
    0: 'follow',
    1: 'friends',
    2: 'help',
    3: 'hurt',
    4: 'my'
}
threshold = 0.8
device = 'cpu'
model = torch.load('trained_model_2_balanced_data_10_epoch.pth')

def save_array_from_video(video_url):
    current_second = 0
    cap = cv2.VideoCapture(video_url)

    frame_interval = 3 
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    num_frames = 10

    frames = []
    list_of_saved_files = {}
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            for i in range(num_frames):
                ret, frame = cap.read()
                current_second = int(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
                frames.append(np.array(frame))
        else:
            break
        if cv2.waitKey(int(frame_interval * frame_rate)) & 0xFF == ord('q'):
            break
        if current_second % frame_interval == 0 and current_second != 0:
            frame_save_name = "/home/works/Pictures/data/second_{}.npy".format(current_second)
            np.save(frame_save_name,frames)
            list_of_saved_files[current_second] = frame_save_name
            frames = []

    cap.release()
    cv2.destroyAllWindows()
    return list_of_saved_files


def inference_function(list_of_saved_files):
    for each_key, value in list_of_saved_files.items():
        counter = []
        pred_data = np.load(value)
        for each_frame in range(len(pred_data)):
            frame = pred_data[each_frame]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame,(224, 224))
            frame = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(frame)
            frame = frame.unsqueeze(0)
            frame = frame.to(device)

            with torch.no_grad():
                output = model(frame)
                probs = torch.softmax(output, dim=1)
                _, pred = torch.max(probs, 1)
                if probs[0][pred.item()] > threshold:
                    counter.append(pred.item())

        if counter:
            mode_vote = Counter(counter).most_common()[0][0]
            print("Class label after Maximum voting at {} is {}".format(each_key, classes[mode_vote]))
        else:
            print("no class above threshold.")
    
inference_function(list_of_saved_files)
