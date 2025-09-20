# Median for up sampling and down sampling

import os
import glob
import numpy as np

# num_videos = 9
num_steps = 7

steps_dic = {
    "1.1 Preparation": 1,
    "1.2 CalotTriangleDissection": 2,
    "1.3 ClippingCutting": 3,
    "1.4 GallbladderDissection": 4,
    "1.5 GallbladderPackaging": 5,
    "1.6 CleaningCoagulation": 6,
    "1.7 GallbladderRetraction": 7,    
}


def convert_time_to_sec(time):
    if time != "0":
        h, m, s = str(time).split(":")
        return int(h) * 3600 + int(m) * 60 + int(s)
    else:
        return int(0)


def convert_sec_to_time(sec):
    m, s = divmod(sec, 60)
    return f"0:{int(m)}:{int(s)}"


# list all annotated videos and shuffle them
ANNOTATION_PATH = "D:/Research/Cholec80/1 - New Annotations"
videos = []
for v in glob.glob(f"{ANNOTATION_PATH}/*/"):
    videos.append(v.split("\\")[-2])

print(len(videos))

num_videos = len(videos)
# random.shuffle(videos)

# calculate statistics, find the median for each step
steps_durations = np.zeros((num_videos, num_steps))

for index, video in enumerate(videos):
    print('index=',index,'video=',video)
    with open(f"{ANNOTATION_PATH}/{video}/GJ.txt", "r") as f:
        lines = f.readlines()
        video_length = convert_time_to_sec(
            lines[-1].split(" : ")[1].split("(")[1].split(",")[1].split(")")[0]
        )

        # print(video_length)

        for step in steps_dic:
            # print(step)
            if step != "Nothing":
                # print(step)
                step_duration = 0
                for line in lines:
                    # print(line.split(" : ")[0])
                    # print(line)
                    if step == line.split(" : ")[0]:
                        b = convert_time_to_sec(
                            line.split(" : ")[1].split("(")[1].split(",")[0]
                        )
                        e = convert_time_to_sec(
                            line.split(" : ")[1]
                            .split("(")[1]
                            .split(",")[1]
                            .split(")")[0]
                        )

                        step_duration += int(e) - int(b)

                steps_durations[index, int(steps_dic[step])-1] = step_duration
                # print(step_duration)
        
        # sum_steps_duration = np.sum(steps_durations[index, :])
        # steps_durations[index, 0] = video_length - sum_steps_duration
        # print(f"{ANNOTATION_PATH}/{video}/GJ.txt", ":", steps_durations[index])

print(steps_durations)

print(np.median(steps_durations, axis=0))
print('median:',np.median(np.median(steps_durations, axis=0)))
# print(np.count_nonzero(steps_durations, axis=0))

# median of all steps in all videos = 259.5