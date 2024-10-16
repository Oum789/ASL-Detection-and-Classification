import cv2
from ultralytics import YOLO
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import os

class_names = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
    'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'
]

# class_names = [
#     '0','1','2','3','4','5','6','7','8','9',
#     'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
#     'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
#     'U', 'V', 'W', 'X', 'Y', 'Z'
# ]



# model_path = os.path.join('8m_2022_100.pt')
model_path = os.path.join('8l_1742_50_Roboflow.pt')

# Load a YOLO model for hand detection
yolo_model = YOLO(model_path).to('cuda')  # Load a custom YOLO model

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the pre-trained MobileNet with the same architecture as the saved model
model = models.mobilenet_v2(pretrained=False)  # pretrained=False since we're loading custom weights
num_classes = 29  # Update with your number of classes
model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)

# Load the saved model weights
model.load_state_dict(torch.load('sign_language_model_100_0_0.pth'))

# Move the model to the appropriate device
model.to(device)

# Set model to evaluation mode (important for inference)
model.eval()

# Define the transform (same as used during training)
mobilenet_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Same normalization as during training
])

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)  # Use 0 for default webcam

while True:
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        break
    
    # Run YOLO hand detection on the current frame
    results = yolo_model(frame)

    # Loop over detected hands (assuming one hand detection for now)
    for result in results:
        for box in result.boxes:
            # Extract bounding box coordinates
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
            
            # Crop the hand region from the frame
            hand_crop = frame[y_min:y_max, x_min:x_max]
            
            # Preprocess the cropped hand image using MobileNet-specific transforms
            hand_image = mobilenet_transform(hand_crop).unsqueeze(0).to('cuda')  # Add batch dimension

            # label = f'{hand_image.shape}'

            # Pass the preprocessed hand image to the MobileNet model for classification
            with torch.no_grad():  # Disable gradient calculation for inference
                predictions = model(hand_image)

            # Get the class label from predictions
            _, predicted_class = predictions.max(1)

            # Map the predicted class index to the class name
            predicted_label = class_names[predicted_class.item()]
            label = f'Class: {predicted_label}'

            # Draw the bounding box and label on the original frame
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the frame with detections and classifications
    cv2.imshow('Hand Detection and Classification', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
