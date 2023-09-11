import torch
import cv2, torch
import argparse
import numpy as np

im_path = './assets/notredame.jpg'

def parseArg():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="input image"
    , default=im_path) 

    args = parser.parse_args()
    return args

# main
if __name__ == "__main__":
    args = parseArg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    og_img = cv2.imread(args.input)
    img = cv2.cvtColor(og_img, cv2.COLOR_BGR2GRAY)[..., np.newaxis]

    detector = torch.hub.load("verlab/LearningToDetect_PRL_2023:main", "Detector", pretrained=True, force_reload=True)

    detector.to(device)
    detector.eval()

    print(f"Img shape: {img.shape}")

    score_map, keypoints, feats = detector.detect(img, 1024)
    
    print("Number of keypoints: ", len(keypoints))

    # plot keypoints
    for kp in keypoints:
        cv2.circle(og_img, (int(kp[0]), int(kp[1])), 3, (255, 0, 0), -1)
    
    cv2.imwrite("output.png", og_img)

    print("Done!")
