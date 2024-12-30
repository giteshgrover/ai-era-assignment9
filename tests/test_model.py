import torch
import pytest
from src.models.resnet50 import create_resnet50


def test_model_training():
    model = create_resnet50()
    test_input = torch.randn(4, 3, 224, 224)
    test_target = torch.randint(0, 10, (4,))
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Test if model can perform one training step
    output = model(test_input)
    loss = criterion(output, test_target)
    loss.backward()
    optimizer.step() 