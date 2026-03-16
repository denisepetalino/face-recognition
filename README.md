# Face Recognition and Attendance Project
This repository is a learning implementation built by following the YouTube tutorial below, which follows the Medium article below. The code and approach in this repo reflect the tutorial flow and structure.
- YouTube tutorial (followed step by step):
  https://www.youtube.com/watch?v=sz25xxF_AVE&t=1201s
- Reference article (Machine Learning is Fun! Part 4: Modern Face Recognition with Deep Learning):
  https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78

### Overview
This project demonstrates two face recognition workflows using Python:

- **Basic face comparison**: compare a known image against a test image and display the match result and distance.
- **Webcam attendance**: encode known faces from a folder, recognise faces from a live webcam feed and write attendance entries to a CSV file.

### Repository structure
- `Basics.py`  
  Loads two images from `ImagesBasic/`, encodes faces, compares them, then displays a match label and distance.
- `AttendanceProject.py`  
  Loads images from `ImagesAttendance/`, builds known encodings, recognises faces via webcam, then records attendance to `Attendance.csv`.
- `ImagesBasic/`  
  Example images used for a simple one to one comparison.
- `ImagesAttendance/`  
  Known faces. The filename (without extension) is used as the person’s name.
- `Attendance.csv`  
  Stores attendance records in `NAME,DATE_TIME` format.

### How to run
Install dependencies: `pip install opencv-python numpy face-recognition`

`Basics.py`
1. Place images in `ImagesBasic`
2. Update filenames if necessary
3. Run `python Basics.py`

`AttendanceProject.py`
1. Add one clear image per person into `ImagesAttendance`
2. Name the file accordingly e.g., an image of Elon Musk `Elon Musk.png`
3. Run `python AttendanceProject.py`

### Possible Improvements
- Use an existing dataset (e.g, Kaggle) rather than local images
- Add a quit option, release camera properly and create `Attendance.csv` if missing
- Require a match across multiple frames to reduce false positives
- Allow multiple images per person and improve matching
- Failure modes: lighting, pose, webcam quality, lookalikes, threshold
