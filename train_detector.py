from ultralytics import YOLO

def train_model():
    
    model = YOLO('yolov8m.pt').to("cuda") 
    
    results = model.train(data="dataset2022.yaml", epochs=100, device="cuda")

if __name__ == '__main__':
    train_model()

