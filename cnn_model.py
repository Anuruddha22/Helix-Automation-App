import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import torchvision.transforms as transforms

class EfficientNetSegmentation(nn.Module):
    def __init__(self):
        super(EfficientNetSegmentation, self).__init__()
        
        # Load pre-trained EfficientNet (efficientnet-b0) for feature extraction
        self.features = EfficientNet.from_pretrained('efficientnet-b0')

        # Optionally, freeze the feature extractor if needed (currently set to trainable)
        for param in self.features.parameters():
            param.requires_grad = True  # Set to False if you want to freeze these layers

        # Upsampling layers to convert extracted features into a segmentation mask
        self.upsample = nn.Sequential(
            # First upsampling step, using ConvTranspose2d to increase spatial resolution
            nn.ConvTranspose2d(1280, 256, kernel_size=4, stride=4),  # 1280 is the feature size from EfficientNet-b0
            nn.SiLU(),  # Using SiLU activation function (equivalent to Swish for better non-linearity)
            
            # Second upsampling step to further increase spatial resolution
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=4),
            nn.SiLU(),

            # Final upsampling step to reduce to the desired number of output channels (1 for binary segmentation)
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2),
            
            # Sigmoid activation for the mask output to constrain values between 0 and 1
            nn.Sigmoid()  # Suitable for binary mask output (use Softmax if multi-class segmentation is needed)
        )

        # Additional head for predicting bounding box coordinates
        self.box_head = nn.Sequential(
            nn.Flatten(),  # Flatten the feature map into a single vector for fully connected layers
            nn.Linear(1280, 512),  # First fully connected layer with 1280 input features
            nn.SiLU(),  # SiLU activation function
            nn.Linear(512, 4)  # Output 4 values representing the bounding box coordinates (x1, y1, x2, y2)
        )
    
    def forward(self, x):
        # Extract features using EfficientNet's feature extractor
        features = self.features.extract_features(x)
        
        # Segmentation head: Upsample the extracted features to create the mask
        mask = self.upsample(features)
        
        # Bounding box prediction head:
        # Global average pooling on spatial dimensions (height and width) to reduce the feature map
        box = self.box_head(features.mean([2, 3]))  # Use mean pooling to get a feature vector per image
        
        # Return both the segmentation mask and the bounding box prediction
        return mask, box
    
if __name__ == "__main__":

    # initizalize CNN model
    model = EfficientNetSegmentation()

    # Define the image transformation pipeline
    transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert the input image (tensor or array) to a PIL image for easier resizing
        transforms.Resize((224, 224)),  # Resize the image to 224x224, the input size expected by EfficientNet
        transforms.ToTensor(),  # Convert the image back to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet pre-trained values
    ])

    # Load the saved model weights from the specified path
    model.load_state_dict(torch.load('HelixApp/cnn_model.pth', map_location=torch.device('cpu')))
    # 'map_location=torch.device("cpu")' ensures the model is loaded on the CPU, useful if you don't have access to a GPU

    # Set the model to evaluation mode, which is crucial when making predictions
    # This disables dropout and batch normalization behaviors that are different during training
    model.eval()