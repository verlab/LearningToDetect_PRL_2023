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

    img = cv2.imread(args.input)

    detector = torch.hub.load("verlab/LearningToDetect_PRL_2023:main", "Detector", pretrained=True, force_reload=True)

    detector.to(device)
    detector.eval()

    print(f"Img shape: {img.shape}")

    keypoints, score_map = detector.detect(img, 1024)

    print("Score Map:", score_map.shape)
    print("Keypoints:", keypoints.shape)

    import pdb; pdb.set_trace()

    # plot keypoints
    for kp in keypoints:
        cv2.circle(img, (int(kp[0]), int(kp[1])), 3, (255, 0, 0), -1)
    
    cv2.imwrite("output.png", img)

    print("Done!")
