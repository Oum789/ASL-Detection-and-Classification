from ultralytics import YOLO

# Define your training script in a function
def train_model():
    # Load the model
    model = YOLO('yolov8m.pt').to("cuda")  # Or another YOLOv8 model
    
    # Train the model
    results = model.train(data="dataset2022.yaml", epochs=100, device="cuda")

# Ensure proper entry point
if __name__ == '__main__':
    train_model()

