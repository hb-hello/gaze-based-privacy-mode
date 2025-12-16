# gaze-based-privacy-mode
Implementation of facial landmark detection and gaze analysis to intelligently manage computer access based on user enggement

Code files 

- `app.py` : This contains the UI logic of the app, i.e. all the Tkinter code as well the decision logic (e.g., when attention dips below threshold for two frames show screen locked)
- `gaze_detect.py` : Gaze parsing logic of the app, i.e. leverages external libraries and image processing systems to generate an attention percentage for each frame captured. This is called every 10th frame and has calculation logic of the EAR which is used.
- `assets` : Directory with all icons and logos

Credit to the open source gaze tracking library developed by antoinelame - https://github.com/antoinelame/GazeTracking
