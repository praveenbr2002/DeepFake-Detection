import torch
import torch.nn as nn
import torchvision.models as models
import timm

class EfficientNetB1LSTM(nn.Module):

    def __init__(self, input_size=128, hidden_size=512, num_layers=2, num_classes=1):
        super(EfficientNetB1LSTM, self).__init__()
        self.b1 = timm.create_model('efficientnet_b1', pretrained=True)

        self.b1 = nn.Sequential(*list(self.b1.children())[:-2],
                    nn.Conv2d(1280, 128, 1, bias=False),
                    nn.BatchNorm2d(128),
                    timm.models.efficientnet.Swish(),
                    nn.AdaptiveAvgPool2d((1, 1)))
        
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):

        batch_size, num_frames, channels, height, width = x.size()

        c_in = x.reshape(batch_size * num_frames, channels, height, width)
        c_out = self.b1(c_in)

        c_out = c_out.view(batch_size, num_frames, -1)
        r_out, _ = self.lstm(c_out)

        out = self.relu(self.fc1(r_out[:, -1, :]))

        result = self.fc2(out)
        return result