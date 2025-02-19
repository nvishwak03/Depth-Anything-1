import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'])
    args = parser.parse_args()

    margin_width = 50
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    depth_anything = DepthAnything.from_pretrained(f'LiheYoung/depth_anything_{args.encoder}14').to(DEVICE).eval()
    
    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    # Open webcam
    cap = cv2.VideoCapture(0)  # Use default webcam (change index if multiple webcams)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_width = frame_width * 2 + margin_width
    
    while True:
        ret, raw_frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB) / 255.0
        frame = transform({'image': frame})['image']
        frame = torch.from_numpy(frame).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            depth = depth_anything(frame)

        depth = F.interpolate(depth[None], (frame_height, frame_width), mode='bilinear', align_corners=False)[0, 0]
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.cpu().numpy().astype(np.uint8)
        depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
        
        # Add margin
        split_region = np.ones((frame_height, margin_width, 3), dtype=np.uint8) * 255
        combined_frame = cv2.hconcat([raw_frame, split_region, depth_color])
        
        # Display the frame
        cv2.imshow('Depth Estimation (Press Q to exit)', combined_frame)
        
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
