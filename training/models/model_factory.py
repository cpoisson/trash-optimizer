"""Model factory for creating pre-trained models."""
import torch.nn as nn
from torchvision import models


class ModelFactory:
    """Factory for creating and configuring pre-trained models."""

    SUPPORTED_MODELS = [
        'efficientnet_b0',
        'efficientnet_b2',
        'efficientnet_v2_s',
        'efficientnet_v2_m',
        'convnext_tiny',
        'resnet50'
    ]

    @staticmethod
    def create(model_name, num_classes, freeze_layers=-3):
        """Create a pre-trained model modified for custom number of classes.

        Args:
            model_name: Name of the model architecture.
            num_classes: Number of output classes.
            freeze_layers: Number of layer blocks to freeze from the end (negative value).
                          For example, -3 means freeze all except last 3 blocks.

        Returns:
            torch.nn.Module: Configured model ready for training.

        Raises:
            ValueError: If model_name is not supported.
        """
        if model_name not in ModelFactory.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model: {model_name}. "
                f"Supported models: {', '.join(ModelFactory.SUPPORTED_MODELS)}"
            )

        if model_name == 'efficientnet_b0':
            return ModelFactory._create_efficientnet_b0(num_classes, freeze_layers)
        elif model_name == 'efficientnet_b2':
            return ModelFactory._create_efficientnet_b2(num_classes, freeze_layers)
        elif model_name == 'efficientnet_v2_s':
            return ModelFactory._create_efficientnet_v2_s(num_classes, freeze_layers)
        elif model_name == 'efficientnet_v2_m':
            return ModelFactory._create_efficientnet_v2_m(num_classes, freeze_layers)
        elif model_name == 'convnext_tiny':
            return ModelFactory._create_convnext_tiny(num_classes, freeze_layers)
        elif model_name == 'resnet50':
            return ModelFactory._create_resnet50(num_classes, freeze_layers)

    @staticmethod
    def _create_efficientnet_b0(num_classes, freeze_layers):
        model = models.efficientnet_b0(weights='IMAGENET1K_V1')
        if freeze_layers < 0:
            for param in model.features[:freeze_layers].parameters():
                param.requires_grad = False
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    @staticmethod
    def _create_efficientnet_b2(num_classes, freeze_layers):
        model = models.efficientnet_b2(weights='IMAGENET1K_V1')
        if freeze_layers < 0:
            for param in model.features[:freeze_layers].parameters():
                param.requires_grad = False
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    @staticmethod
    def _create_efficientnet_v2_s(num_classes, freeze_layers):
        model = models.efficientnet_v2_s(weights='IMAGENET1K_V1')
        if freeze_layers < 0:
            for param in model.features[:freeze_layers].parameters():
                param.requires_grad = False
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    @staticmethod
    def _create_efficientnet_v2_m(num_classes, freeze_layers):
        model = models.efficientnet_v2_m(weights='IMAGENET1K_V1')
        if freeze_layers < 0:
            for param in model.features[:freeze_layers].parameters():
                param.requires_grad = False
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    @staticmethod
    def _create_convnext_tiny(num_classes, freeze_layers):
        model = models.convnext_tiny(weights='IMAGENET1K_V1')
        if freeze_layers < 0:
            # ConvNeXt uses features[:-2] pattern
            freeze_idx = max(0, len(model.features) + freeze_layers + 1)
            for param in model.features[:freeze_idx].parameters():
                param.requires_grad = False
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
        return model

    @staticmethod
    def _create_resnet50(num_classes, freeze_layers):
        model = models.resnet50(weights='IMAGENET1K_V2')
        # For ResNet, freeze early layers (layer1, layer2)
        if freeze_layers < 0:
            for param in model.layer1.parameters():
                param.requires_grad = False
            for param in model.layer2.parameters():
                param.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
