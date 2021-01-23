# This program converts a video into frames at the specified frequency

# Desired input: .mp4 video from Android
# Desired output: frames per second

import cv2 as cv

# Captures frame_test.mp4 and assigns it to cap
cap = cv.VideoCapture('C:/Users/Jared/Desktop/ECEN 404/hello2/videos/detect_train.mp4')

count = 1 # Number of total frames
freq = 5 # Save an image every "freq" frames
file = 1 # File numbering variable

# While the video is being processed
while cap.isOpened():
    fileStr = str(file)
    ret, frame = cap.read()
    if not ret:
        print("Frame unreadable or end of frames")
        break
    if count % freq == 0:
        cv.imwrite('frames/'+fileStr+'.jpg',frame)
        file = file + 1;
    if cv.waitKey(1) == ord('q'):
        break
    count = count + 1

cap.release()
cv.destroyAllWindows()
print("Total frames:",count-1)
print("Frames saved:",file-1)
