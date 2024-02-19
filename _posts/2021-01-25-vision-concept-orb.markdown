---
layout: post
title: ORB (Oriented FAST and Rotated BRIEF)
date: 2021-01-25 03:00:00
img: vision/concept/orb/0.png
categories: [vision-concept] 
tags: [vision, concept, feature extraction, ORB] # add tag
---

<br>

[Vision 관련 글 목차](https://gaussian37.github.io/vision-concept-table/)

<br>

- 논문 : https://www.gwylab.com/download/ORB_2012.pdf
- 참조 : https://medium.com/data-breach/introduction-to-orb-oriented-fast-and-rotated-brief-4220e8ec40cf

<br>

- 아래는 동영상 파일을 읽어서 `ORB`를 적용한 후 시각화한 예시 코드 입니다.

<br>

```python
# Open the video
cap = cv2.VideoCapture('./test.mp4')

# Read the first frame
ret, prev_frame = cap.read()

H, W = prev_frame.shape[0], prev_frame.shape[1]
prev_frame = cv2.resize(prev_frame, (W//2, H//2))

# Convert the first frame to grayscale
prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Initialize the ORB detector
orb = cv2.ORB_create(nfeatures=200)

# Create a BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

paused = False
num_points = 50

while True:
    if not paused:        
        # Read the next frame
        ret, curr_frame = cap.read()
        if not ret:
            break  # Break the loop if there are no more frames
        
        # Convert current frame to grayscale
        curr_frame = cv2.resize(curr_frame, (W//2, H//2))
        curr_frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Detect keypoints and compute descriptors for both frames
        keypoints_prev, descriptors_prev = orb.detectAndCompute(prev_frame_gray, None)
        keypoints_curr, descriptors_curr = orb.detectAndCompute(curr_frame_gray, None)

        if descriptors_prev is None or descriptors_curr is None:
            # Prepare for the next iteration
            prev_frame = curr_frame
            prev_frame_gray = curr_frame_gray
            continue
        
        # Match descriptors
        matches = bf.match(descriptors_prev, descriptors_curr)
        
        # Sort them in the order of their distance
        matches = sorted(matches, key=lambda x:x.distance)
        
        # Draw first 10 matches
        matched_image = cv2.drawMatches(prev_frame, keypoints_prev, curr_frame, keypoints_curr, matches[:num_points], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
    # Display the matched image
    cv2.imshow('Matched Features', matched_image)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):  # Break the loop when 'q' is pressed
        break
    elif key == ord('p'):
        paused = not paused  # Toggle pause
    
    # Prepare for the next iteration
    prev_frame = curr_frame
    prev_frame_gray = curr_frame_gray

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
```

<br>

[Vision 관련 글 목차](https://gaussian37.github.io/vision-concept-table/)

<br>
