from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# from pytorch_grad_cam.utils.image import show_cam_on_image

# input_tensor can be a mini batch
def grad_cam(input_tensor, model, class_idx):
    target_layers = [model.features[-1]]
    # We have to specify the target we want to generate the CAM for.
    # 14 classes for chexpert dataset
    targets = [ClassifierOutputTarget(class_idx)]

    # Construct the CAM object once, and then re-use it on many images.
    with GradCAM(model=model, target_layers=target_layers) as cam:
        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        grayscale_cam_batch = cam(input_tensor=input_tensor, targets=targets)
        #   visualization = show_cam_on_image(input_img, grayscale_cam, use_rgb=True)
        # You can also get the model outputs without having to redo inference
        #   model_outputs = cam.outputs

    return grayscale_cam_batch