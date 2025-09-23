import cv2

def find_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at (x, y): ({x}, {y})")

# --- Main Part ---
image_path = 'data/color_correction/input2.bmp'
img = cv2.imread(image_path)

if img is None:
    print(f"Error: Could not load image at {image_path}")
else:
    cv2.imshow('Image - Click to find coordinates', img)
    cv2.setMouseCallback('Image - Click to find coordinates', find_coordinates)
    
    print("Click on the corners of the white patch. Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()