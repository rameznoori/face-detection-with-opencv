
# Face Detection with OpenCV and Caffe

This Python project performs real-time face detection using OpenCV’s deep learning module with a pre-trained Caffe model. By default, it uses a **webcam or video stream to detect faces frame-by-frame**, drawing bounding boxes and confidence scores. The project also supports **image-based face detection**, which can be enabled by commenting out the video stream code and uncommenting the image processing section. This dual functionality makes it versatile for both live video and static image face detection tasks.
## Features

- Real-time face detection from webcam or video source
- Detects faces using a Caffe deep learning model with configurable confidence threshold
- Displays bounding boxes and confidence percentages on detected faces
- Option to switch to static image face detection by modifying comments in the script
- Resizes frames for optimized processing

## Getting Started / Installation

- Ensure Python 3.7 or later is installed.
- Install required libraries via pip:
    ```
    pip3 install opencv-python imutils numpy
    ```
- Download the *Caffe deploy prototxt* and *pre-trained model* files.
- Run the script from the command line with required arguments:
    - For live video face detection:
        ```
            python3 face_detection.py --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel

        ```
        - The webcam face detection window will open, press `q` to quit.
    - For image-based face detection:
        ```
            python3 face_detection.py --image rooster.jpg --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel
        ```

## Usage Instructions

- **Video/Webcam Detection (default):** Runs real-time face detection on video frames. Launch the script with prototxt and model paths as shown above.
- **Image Detection:** To detect faces in a static image:
	- Comment out the video stream and loop code block.
	- Uncomment the image reading and detection code block near the bottom of the script.
	- Run the script with the `--image` argument specifying the path to your input image. (As described above)
    - Detected faces with bounding boxes and confidence scores will be shown in an image window.
## Technologies Used

- **Python 3.x**
- **OpenCV** for computer vision operations and DNN module
- **Caffe model** and prototxt files for face detection
- **imutils** for frame resizing and video stream handling
- **numpy** for array manipulation
## File Structure

```
/
├── face_detection.py          # Main Python script for face detection logic
├── deploy.prototxt.txt        # Caffe deploy prototxt file for model architecture
├── res10_300x300_ssd_iter_140000.caffemodel  # Pre-trained Caffe model weights
├── README.md                  # This documentation file
├── rooster.jpg                # A sample image to test the image-based face detection
├── screenshot.png             # Live video face detection sample output
├── Image-Screenshot.png       # Image based face detection sample output

```
## Customization

- Adjust the `--confidence` argument value to change the detection confidence threshold (default 0.5).
- Modify frame resizing dimensions (currently width=800) for different video processing speeds and qualities.
- Extend to support video file input instead of webcam by modifying the video stream source.
- Integrate different face detection models or frameworks.
## Known Issues / Limitations

- Requires compatible webcam device for live video detection.
- Image mode requires manual comment/uncomment of code blocks.
- Detection is limited by the accuracy and speed of the used Caffe model.
- No GUI controls, only keyboard interaction (press `q` to quit video window).
## Future Improvements

- Add command-line flags to switch between image and video detection without code modification.
- Implement GUI interface with buttons to toggle modes.
- Add support for multiple video sources or recorded videos.
- Incorporate other detection models (e.g., TensorFlow, ONNX).
- Optimize speed using GPU acceleration where available.
## Contributing

Contributions are always welcome!

- Fork the repository.
- Create a feature or bugfix branch.
- Document your code and test thoroughly.
- Submit pull requests with clear explanations.
## License

This project is open-source under the MIT License.
## Acknowledgements

 - OpenCV community and contributors for computer vision tools.
- Authors of the Caffe face detection model.
- imutils development team for streamlining video processing.