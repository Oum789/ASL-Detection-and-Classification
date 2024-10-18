import cv2
from ultralytics import YOLO
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import os
import torch.nn.functional as F  

class_names = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
    'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'
]

model_path = os.path.join('8l_1742_50_Roboflow.pt')

yolo_model = YOLO(model_path) 

model = models.mobilenet_v2(pretrained=False)  
num_classes = 29  
model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)

model.load_state_dict(torch.load('sign_language_model_epoch_19.pth'))
model.eval()

mobilenet_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

cap = cv2.VideoCapture(0) 

while True:
    ret, frame = cap.read() 
    if not ret:
        break
    
    results = yolo_model(frame)

    for result in results:
        for box in result.boxes:
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
            yolo_confidence = float(box.conf[0])  
            
            hand_crop = frame[y_min:y_max, x_min:x_max]
            hand_image = mobilenet_transform(hand_crop).unsqueeze(0)

            with torch.no_grad(): 
                predictions = model(hand_image)
                softmax_preds = F.softmax(predictions, dim=1)  
                mobile_confidence, predicted_class = torch.max(softmax_preds, 1)

            predicted_label = class_names[predicted_class.item()]
            yolo_class_label = class_names[int(box.cls[0])]
            label = (
                f'{yolo_class_label} ({yolo_confidence:.2f}), '
                f'{predicted_label} ({mobile_confidence.item():.2f})'
            )

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow('Hand Detection and Classification', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
