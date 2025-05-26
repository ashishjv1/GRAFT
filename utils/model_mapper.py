from models import ResNet18,ResNet50, EfficientNetB0, ResNet9, MobileNetV2, MobileNet, ResNet101, ResNet152, ResNeXt29_32x4d, ResNext50_32x4d, ResNext101_32x8d, ResNext101_64x4d, FashionCNN, bertmodel


class ModelMapper:
    def __init__(self, args):
        self.args = args
        self.model_mapping = {
            "resnet18": ResNet18,
            "resnet50": ResNet50,
            "resnet9": ResNet9,
            "mobilenetv2": MobileNetV2,
            "mobilenet": MobileNet,
            "resnet101": ResNet101,
            "resnet152": ResNet152,
            "efficientnetb0": EfficientNetB0,
            "resnext": ResNeXt29_32x4d,
            "resnext50": ResNext50_32x4d,
            "resnext101_32": ResNext101_32x8d,
            "resnext101_64": ResNext101_64x4d,
            "fashioncnn": FashionCNN,
#             "twolayernet": TwoLayerNet,
#             "threelayernet": ThreeLayerNet,
            "bert": bertmodel,
        }

        
    def get_model(self):
        # Get the model name from arguments and convert it to lowercase
        
        model_group1 = ["resnext", "resnet9", "fashioncnn", "twolayernet", "threelayernet"]
        model_group2 = ["bert"]
        
        model_name = self.args.model.lower()
        if model_name in self.model_mapping:
            # Get the corresponding model class
            model_class = self.model_mapping[model_name]
            if model_name.lower() in model_group1:
                return model_class(self.args.in_chanls, self.args.numClasses).to(self.args.device)
            elif model_name.lower() in model_group2:
                return model_class(self.args.device, self.args.numClasses).to(self.args.device)
            else:
                return model_class(self.args.numClasses).to(self.args.device)
        else:
            print("model not available at this moment")
            