"""
Evaluate trained models on the official CUB test set
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import sys
import torch
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from CUB.dataset import load_data
from CUB.config import BASE_DIR, N_CLASSES, N_ATTRIBUTES
from captum.attr import IntegratedGradients
# from captum.attr import LayerConductance
# from captum.attr import NeuronConductance

class LabelPredictor(torch.nn.Module):
    def __init__(self, model1, model2):
        super(LabelPredictor, self).__init__()
        self.first_model = model1
        self.sec_model = model2

    def forward(self, x):
        attr_outputs = torch.cat(self.first_model(x), dim=1)
        return self.sec_model(attr_outputs)

class ConceptPredictor(torch.nn.Module):
    def __init__(self, model1):
        super(ConceptPredictor, self).__init__()
        self.conceptModel = model1

    def forward(self, x):
        return torch.cat(self.conceptModel(x), dim=1)
        
# def visualize_attribution(inputs, attributions, className, predictClass, path=None):
#     original_image = inputs.cpu().detach().numpy().squeeze().transpose(1, 2, 0)
#     attribution_image = attributions.cpu().detach().numpy().squeeze().transpose(1, 2, 0)

#     original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())
#     attribution_image = (attribution_image - attribution_image.min()) / (attribution_image.max() - attribution_image.min())
#     threshold = 0.5  # Set a threshold value (you can adjust it)
#     attribution_image[attribution_image < threshold] = 0 

#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
#     ax1.imshow(original_image)
#     ax1.axis('off')
#     ax1.set_title(f'Original {className}')
#     heatmap = ax2.imshow(attribution_image, cmap='inferno')
#     ax2.axis('off')
#     ax2.set_title(f'Predicted {predictClass}')
#     fig.colorbar(heatmap, ax=ax2, orientation='vertical')
#     plt.savefig(path)

def visualize_attribution(inputs, label_attributions, concept_attributions, concepts_names, concept_predictions, className, predictClass, path=None):
    original_image = inputs.cpu().detach().numpy().squeeze().transpose(1, 2, 0)
    original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())

    label_threshold = 0.03 #keep top k percent
    label_attribution_image = label_attributions.cpu().detach().numpy().squeeze().transpose(1, 2, 0)
    label_attribution_image = (label_attribution_image - label_attribution_image.min()) / (label_attribution_image.max() - label_attribution_image.min())
    label_threshold_value = np.percentile(label_attribution_image, (1 - label_threshold) * 100)
    label_attribution_image[label_attribution_image < label_threshold_value] = 0
    label_attribution_image[label_attribution_image >= label_threshold] = 1

    concept_threshold = 0.001 #keep top k percent
    processed_concept_attributions = []
    for concept_attribution in concept_attributions:
        concept_image = concept_attribution.cpu().detach().numpy().squeeze().transpose(1, 2, 0)
        concept_image = (concept_image - concept_image.min()) / (concept_image.max() - concept_image.min())
        concept_threshold_value = np.percentile(concept_image, (1 - concept_threshold) * 100)  # Top p
        concept_image[concept_image < concept_threshold_value] = 0
        concept_image[concept_image >= concept_threshold] = 1
        processed_concept_attributions.append(concept_image)

    # Sort concept predictions and corresponding concept attributions and names
    sorted_indices = torch.argsort(concept_predictions, descending=True)  # Sort indices by predictions (high to low)
    concept_predictions = concept_predictions[sorted_indices]
    processed_concept_attributions = [processed_concept_attributions[i] for i in sorted_indices]
    concepts_names = [concepts_names[i] for i in sorted_indices]

    total_images = 2 + len(concept_attributions)
    fig, axes = plt.subplots(1, total_images, figsize=(10+5*len(concept_attributions), 5))

    # Plot original image
    axes[0].imshow(original_image)
    axes[0].axis('off')
    axes[0].set_title(f'{className}')

    # Plot label attribution
    heatmap_label = axes[1].imshow(label_attribution_image)
    axes[1].axis('off')
    axes[1].set_title(f'Predicted: {predictClass}')
    
    # Plot concept attributions
    for idx, concept_image in enumerate(processed_concept_attributions):
        heatmap_concept = axes[idx + 2].imshow(concept_image)
        axes[idx + 2].axis('off')
        axes[idx + 2].set_title(f'{concepts_names[idx]}: {concept_predictions[idx].item():.2f}')

    plt.savefig(path)
        
def run_feature_attribution(args):
    classes_name = pd.read_csv("/home/konghaoz/cbm/CUB_200_2011/classes.txt", sep="\s+", header=None, names=["Index", "Name"])
    concepts_name = pd.read_csv("/home/konghaoz/cbm/CUB_200_2011/attributes/attributes_filtered.txt", sep="\s+", header=None, names=["Index", "Name"])

    model = torch.load(args.model_dirs)
    model.eval()

    conceptPredictor = ConceptPredictor(model.first_model)                  #wrapper to the original model proposed in the paper
    labelPredictor = LabelPredictor(model.first_model, model.sec_model)     #wrapper to the original model proposed in the paper

    data_dir = os.path.join(BASE_DIR, args.data_dir, args.eval_data + '.pkl')
    loader = load_data([data_dir], args.use_attr, args.no_img, args.batch_size, image_dir=args.image_dir,
                       n_class_attr=args.n_class_attr)

    ig_label = IntegratedGradients(labelPredictor)
    ig_concept = IntegratedGradients(conceptPredictor)

    for data_idx, data in enumerate(loader):
        
        # inputs, labels, attr_labels, path = data
        inputs, labels, attr_labels, path = data
        attr_labels = torch.stack(attr_labels).t() 

        inputs_var = torch.autograd.Variable(inputs).cuda() 
        labels_var = torch.autograd.Variable(labels).cuda() 
        
        concepts_outputs = conceptPredictor(inputs_var)
        label_outputs = labelPredictor(inputs_var)
        prediction_score, pred_label_idx = torch.topk(label_outputs, 1)
        
        if labels.squeeze().item() == pred_label_idx.squeeze().item():
            # label attribution
            className = classes_name["Name"][labels.squeeze().item()]
            predictClass = classes_name["Name"][pred_label_idx.squeeze().item()]
            inputs_var.requires_grad_()
            labelAttr, delta = ig_label.attribute(inputs_var, target = pred_label_idx, return_convergence_delta=True)

            # concepts attribution
            concepts_indices_list = torch.nonzero(attr_labels == 1, as_tuple=True)[1].tolist()
            concepts_names = concepts_name["Name"][concepts_indices_list].tolist()
            concepts_attributions = []

            concepts_predictions = concepts_outputs[0][concepts_indices_list] #normalize
            min_val = concepts_predictions.min()
            max_val = concepts_predictions.max()    
            concepts_predictions = (concepts_predictions - min_val) / (max_val - min_val)

            for index in concepts_indices_list:
                inputs_var.requires_grad_()
                conceptAttr, delta = ig_concept.attribute(inputs_var, target = torch.tensor([[index]]), return_convergence_delta=True)
                concepts_attributions.append(conceptAttr[0])

            visualize_attribution(inputs_var[0], labelAttr[0], concepts_attributions, concepts_names, concepts_predictions, className, predictClass, path=f"/home/konghaoz/cbm/CUB/outputs/IG/{path[0].split('/')[-1]}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Concepts Validation')
    parser.add_argument('-model_dirs', default=None, help='where the trained models are saved')
    parser.add_argument('-eval_data', default='test', help='Type of data (train/ val/ test) to be used')
    parser.add_argument('-use_attr', default = True, help='whether to use attributes (FOR COTRAINING ARCHITECTURE ONLY)', action='store_true')
    parser.add_argument('-no_img', help='if included, only use attributes (and not raw imgs) for class prediction', action='store_true')
    parser.add_argument('-image_dir', default='images', help='test image folder to run inference on')
    parser.add_argument('-n_class_attr', type=int, default=2, help='whether attr prediction is a binary or triary classification')
    parser.add_argument('-data_dir', default='CUB_processed/class_attr_data_10', help='directory to the data used for evaluation')
    parser.add_argument('-n_attributes', type=int, default=N_ATTRIBUTES, help='whether to apply bottlenecks to only a few attributes')    
    args = parser.parse_args()
    args.batch_size = 1

    attributions = run_feature_attribution(args)
    # python3 CUB/feature_attribution.py -model_dirs /home/konghaoz/cbm/CUB/outputs/best_model_2.pt