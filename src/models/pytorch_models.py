import torch
import torch.nn as nn
import timm

class CustomMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.1):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

class ModelFactory:
    @staticmethod
    def get_timm_model(model_name, num_classes, pretrained=True):
        return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        
    @staticmethod
    def get_custom_mlp(input_dim, hidden_dims, output_dim):
        return CustomMLP(input_dim, hidden_dims, output_dim)

    @staticmethod
    def get_available_image_models():
        return [
            'resnet50',
            'efficientnet_b0',
            'vit_base_patch16_224',
            'deit_base_patch16_224',
            'swin_tiny_patch4_window7_224'
        ]
