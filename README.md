# expression_analysis
this projects takes video ,after that detect and track face to perform expression analysis
Methodology
Step 1: Preprocessing
Preprocess the video data by converting to frames using OpenCV

Step 2: Face Detection using MTCNN
Use MTCNN (Multi-task Cascaded Convolutional Networks) to detect faces within each frame of the video. MTCNN is a popular face detection model that can accurately locate faces in images and videos.

Step 3: Tracking Faces with Deep SORT
Implement Deep SORT (Simple Online and Realtime Tracking) to track the detected faces over time. Deep SORT assigns a unique ID to each face, allowing you to follow individuals as they move throughout the video.

Step 4: Facial Expression Analysis with DeepFace
Use DeepFace to analyze the facial expressions of tracked faces. DeepFace is a deep learning model that can detect emotions such as happiness, sadness, anger, etc., by analyzing the facial features and expressions.

Step 5: Emotion Labeling and Visualization
After analyzing each frame with DeepFace, label the emotions detected for each tracked face and visualize the results. I used text labeler to print the most prominent expression of person on frame along with its score.
