import cv2, torch
import numpy as np
from models.our_detector import Our
import argparse
im_path = './assets/notredame.jpg'
model_path = './pretrained/our/final_model_1500.pth'


def parseArg():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="input image"
    , default=im_path) 
    parser.add_argument("--model", help="Path to model file"
    , default=model_path) 

    args = parser.parse_args()
    return args

# main
if __name__ == "__main__":
    global args
    args = parseArg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    og_img = cv2.imread(args.input)
    img = cv2.cvtColor(og_img, cv2.COLOR_BGR2GRAY)[..., np.newaxis]

    detector = Our()
    detector.load_state_dict(torch.load(args.model, map_location = detector.device))

    detector.to(device)
    detector.eval()

    print(f"Img shape: {img.shape}")

    keypoints, score_map = detector.detect(img, 1024)
    
    print("Number of keypoints: ", len(keypoints))

    # plot keypoints
    for kp in keypoints:
        cv2.circle(og_img, (int(kp[0]), int(kp[1])), 6, (0, 0, 255), -1)
    
    cv2.imwrite("output.png", og_img)

    np.save('s_map.npy', score_map)
    np.save('kps.npy', keypoints)

    print("Done!")
