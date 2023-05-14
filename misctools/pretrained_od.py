#ref - https://coderzcolumn.com/tutorials/artificial-intelligence/object-detection-using-pre-trained-pytorch-models

import torch
import torchvision
#pip install pycocotools
import pycocotools
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from torchvision.models.detection import fasterrcnn_resnet50_fpn #160 MB model
from torchvision.models.detection import retinanet_resnet50_fpn #130 MB model
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn #75 MB model
from torchvision.models.detection import ssdlite320_mobilenet_v3_large #14 MB model

from pycocotools.coco import COCO
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image

import cv2
import numpy as np

annFile='annotations/instances_val2017.json'

coco=COCO(annFile)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def predict_image(pil_image, save_preds = 'output_preds_indoor', idx = 0):
	print("predicting image number ",idx)
	kids_playing = pil_image
	kids_playing_tensor_int = pil_to_tensor(kids_playing)
	kids_playing_tensor_int = kids_playing_tensor_int.unsqueeze(dim=0)
	kids_playing_tensor_float = kids_playing_tensor_int / 255.0
	object_detection_model = retinanet_resnet50_fpn(pretrained=True, progress=False)

	print("number of model parameters ",count_parameters(object_detection_model))

	object_detection_model.eval();

	kids_preds = object_detection_model(kids_playing_tensor_float)


	kids_preds[0]["boxes"] = kids_preds[0]["boxes"][kids_preds[0]["scores"] > 0.5]
	kids_preds[0]["labels"] = kids_preds[0]["labels"][kids_preds[0]["scores"] > 0.5]
	kids_preds[0]["scores"] = kids_preds[0]["scores"][kids_preds[0]["scores"] > 0.5]



	kids_labels = coco.loadCats(kids_preds[0]["labels"].numpy())



	kids_annot_labels = ["{}-{:.2f}".format(label["name"], prob) for label, prob in zip(kids_labels, kids_preds[0]["scores"].detach().numpy())]

	kids_output = draw_bounding_boxes(image=kids_playing_tensor_int[0],
	                             boxes=kids_preds[0]["boxes"],
	                             labels=kids_annot_labels,
	                             colors=["red" if label["name"]=="person" else "green" for label in kids_labels],
	                             width=2,
	                             font_size=16,
	                             fill=True
	                            )

	img = np.array(to_pil_image(kids_output))
	cv2.imshow("output ",np.array(img[...,::-1]))
	cv2.waitKey(10)
	cv2.imwrite(save_preds+'/'+str(idx)+'.png', np.array(img[...,::-1]))


if __name__ == '__main__':
	for i in range(0,500):
		#kids_playing = Image.open("kids-playing.jpg")
		#kids_playing = Image.open("folder_prediction_outdoor/pred"+str(i)+".png")
		kids_playing = Image.open("folder_prediction_indoor/pred"+str(i)+".png")
		predict_image(kids_playing, idx = i)
	






# kids_playing = Image.open("kids-playing.jpg")
# kids_playing_tensor_int = pil_to_tensor(kids_playing)
# kids_playing_tensor_int = kids_playing_tensor_int.unsqueeze(dim=0)
# kids_playing_tensor_float = kids_playing_tensor_int / 255.0


# #object_detection_model = fasterrcnn_resnet50_fpn(pretrained=True, progress=False)
# #object_detection_model = retinanet_resnet50_fpn(pretrained=True, progress=False)
# #object_detection_model = ssdlite320_mobilenet_v3_large(pretrained=True, progress=False)
# object_detection_model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True, progress=False)

# print("number of model parameters ",count_parameters(object_detection_model))

# object_detection_model.eval();

# kids_preds = object_detection_model(kids_playing_tensor_float)


# kids_preds[0]["boxes"] = kids_preds[0]["boxes"][kids_preds[0]["scores"] > 0.8]
# kids_preds[0]["labels"] = kids_preds[0]["labels"][kids_preds[0]["scores"] > 0.8]
# kids_preds[0]["scores"] = kids_preds[0]["scores"][kids_preds[0]["scores"] > 0.8]







# annFile='annotations/instances_val2017.json'

# coco=COCO(annFile)

# kids_labels = coco.loadCats(kids_preds[0]["labels"].numpy())



# kids_annot_labels = ["{}-{:.2f}".format(label["name"], prob) for label, prob in zip(kids_labels, kids_preds[0]["scores"].detach().numpy())]

# kids_output = draw_bounding_boxes(image=kids_playing_tensor_int[0],
#                              boxes=kids_preds[0]["boxes"],
#                              labels=kids_annot_labels,
#                              colors=["red" if label["name"]=="person" else "green" for label in kids_labels],
#                              width=2,
#                              font_size=16,
#                              fill=True
#                             )

# img = np.array(to_pil_image(kids_output))
# cv2.imshow("output ",np.array(img[...,::-1]))
# cv2.waitKey(0)







