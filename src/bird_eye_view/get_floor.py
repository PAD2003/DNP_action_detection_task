import cv2

# Declare a list to store selected points
selected_points = []

def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Pixel coordinates: ({x}, {y})")
        selected_points.append((x, y))

# Load the image
image = cv2.imread("data/raw_image.png")
height, width, channels = image.shape

# Display the image and wait for user to select points
cv2.imshow("Image", image)
cv2.setMouseCallback("Image", mouse_click)

# Loop indefinitely until you are done selecting
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Press the Esc key to exit (ASCII code for Esc is 27)
        break

# Close the window when done
cv2.destroyAllWindows()

# Print the list of selected points
for point in selected_points:
    print(f"Selected pixel coordinates: {(point[0] / width, point[1] / height)}")