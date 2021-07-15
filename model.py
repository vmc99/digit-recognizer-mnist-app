import torch
import torchvision.transforms as T
import torch.nn.functional as F
import torch.nn as nn
import math
import gc
import streamlit as st

#Enable garbage collection
gc.enable()

def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet8(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # 1 x 28 x 28
        self.conv1 = conv_block(in_channels, 64) # 64 x 28 x 28
        self.conv2 = conv_block(64, 128, pool=True) # 128 x 14 x 14
        self.res1 = nn.Sequential(conv_block(128, 128), 
                                  conv_block(128, 128)) # 128 x 14 x 14
        
        self.conv3 = conv_block(128, 256, pool=True)  # 256 x 7 x 7
        self.res2 = nn.Sequential(conv_block(256, 256), 
                                  conv_block(256, 256)) # 256 x 7 x 7
        
        self.classifier = nn.Sequential(nn.MaxPool2d(7),  # 256 x 1 x 1 since maxpool with 7x7
                                        nn.Flatten(),    # 256*1*1 
                                        nn.Dropout(0.2),
                                        nn.Linear(256, num_classes))
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out



def transform_image(image):
    stats = ((0.1307), (0.3081))
    my_transforms = T.Compose([

                        T.ToTensor(),
                        T.Normalize(*stats)

                        ])

    return my_transforms(image)




@st.cache
def initiate_model():

    # Initiate model
    in_channels = 1
    num_classes = 10
    model = ResNet8(in_channels, num_classes)
    device = torch.device('cpu')
    PATH = 'mnist-resnet.pth'
    model.load_state_dict(torch.load(PATH, map_location=device))
    model.eval()

    return model




def predict_image(img):
    
    # Convert to a batch of 1
    xb = img.unsqueeze(0)

    model = initiate_model()

    # Get predictions from model
    yb = model(xb)
    # apply softamx
    yb_soft = F.softmax(yb, dim=1)
    # Pick index with highest probability
    confidence , preds  = torch.max(yb_soft, dim=1)
    gc.collect()
    # Retrieve the class label, confidence and probabilities of all classes using sigmoid 
    return preds[0].item(), math.trunc(confidence.item()*100), torch.sigmoid(yb).detach()