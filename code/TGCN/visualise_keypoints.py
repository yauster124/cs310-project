import json
from pathlib import Path
import matplotlib.pyplot as plt
import cv2

data_root = Path('../../../../../../large/u2008310/data')
pose_path = data_root / 'pose_per_individual_videos'
# keypoint_path = pose_path / '01375/image_00010_keypoints.json'
keypoint_path = 'image_00010_keypoints.json'
# video_path = data_root / "WLASL2000/02854.mp4"
video_path = '02854.mp4'

pose_content = json.load(open(keypoint_path))["people"][0]
body_pose = pose_content["pose_keypoints_2d"]
left_hand_pose = pose_content["hand_left_keypoints_2d"]
right_hand_pose = pose_content["hand_right_keypoints_2d"]

#  0  1  2  3  4  5  6  7   8   9   10
# [0, 1, 2, 3, 5, 6, 8, 15, 16, 17, 18]
# body_pose_exclude = {9, 10, 11, 22, 23, 24, 12, 13, 14, 19, 20, 21}
body_pose_exclude = {4, 7, 9, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24}
body_pose.extend(left_hand_pose)
body_pose.extend(right_hand_pose)

x = [v for i, v in enumerate(body_pose) if i % 3 == 0 and i // 3 not in body_pose_exclude]
y = [v for i, v in enumerate(body_pose) if i % 3 == 1 and i // 3 not in body_pose_exclude]

cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
res, frame = cap.read()
cv2.imwrite('frame.jpg', frame)

im = plt.imread('frame.jpg')
implot = plt.imshow(im)

# put a blue dot at (10, 20)

# put a red dot, size 40, at 2 locations:
plt.scatter(x=x, y=y, s=0.5, color='yellow')

pairs = [
    (0, 1),
    (1, 2),
    (2, 3),
    (1, 4),
    (4, 5),
    (1, 6),
    (0, 8),
    (0, 7),
    (8, 10),
    (7, 9),
    (5, 11),
    (11, 12),
    (12, 13),
    (13, 14),
    (14, 15),
    (11, 16),
    (16, 17),
    (17, 18),
    (18, 19),
    (11, 20),
    (20, 21),
    (21, 22),
    (22, 23),
    (11, 24),
    (24, 25),
    (25, 26),
    (26, 27),
    (11, 28),
    (28, 29),
    (29, 30),
    (30, 31),
    (3, 32),
    (32, 33),
    (33, 34),
    (34, 35),
    (35, 36),
    (32, 37),
    (37, 38),
    (38, 39),
    (39, 40),
    (32, 41),
    (41, 42),
    (42, 43),
    (43, 44),
    (32, 45),
    (45, 46),
    (46, 47),
    (47, 48),
    (32, 49),
    (49, 50),
    (50, 51),
    (51, 52)
]

for a, b in pairs:
    xs = [x[a], x[b]]
    ys = [y[a], y[b]]
    plt.plot(xs, ys, color='red')

plt.show()
plt.savefig('points.png')


    

# plt.scatter(x, y)
# plt.savefig('points.png')

# cap = cv2.VideoCapture('../../../../../../large/u2008310/data/WLASL2000/68105.mp4')
# cap.set(cv2.CAP_PROP_POS_FRAMES, 44)
# res, frame = cap.read()
# cv2.imwrite('frame.jpg', frame)