import cv2

def main():
    # Create a VideoCapture object to access the camera feed
    cap = cv2.VideoCapture(0)  # 0 is the default camera index, change it if needed

    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Loop to continuously read frames from the camera
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        # Check if the frame was read successfully
        if not ret:
            print("Error: Could not read frame.")
            break

        # Display the frame
        cv2.imshow("Camera Feed", frame)

        # Check for the 'q' key to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture object and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
