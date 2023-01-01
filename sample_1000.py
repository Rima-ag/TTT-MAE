# import os
# import shutil

# root = './imagenet_c/'

# dbs = [ f.name for f in os.scandir(root) if f.is_dir() ]

# samples = root + 'samples/'
# os.mkdir(samples)

# for f in dbs:
#     print(f)
#     os.mkdir(samples + f)
#     current_dir = samples + f + '/5/'
#     os.mkdir(current_dir)
#     subfolders = [ s.name for s in os.scandir(root + f + '/5/') if s.is_dir() ]
#     limit = False
#     i = 0
   
#     with open("./test.txt", 'w') as test:
#         test.writelines('rhghqoruh')

#     for sub in subfolders:
#         if not limit:
#             os.mkdir(current_dir + sub)
#             image_parent = root + f + '/5/' + sub
#             for image in os.listdir(image_parent):
#                 if i > 1000:
#                     limit = True
#                     break
                
#                 i += 1
#                 with open(image_parent + '/' + image, 'br') as source:
#                     with open(current_dir + sub + '/' + image, "bw") as write:
#                         print(image_parent + '/' + image)
#                         write.writelines(source.readlines())
#                 print(current_dir + sub + '/' + image)
            

# import numpy as np

# data = np.load('./output/brightness/results_499.npy')

# print(data)

import os
for child in os.listdir('./imagenet_c'):
        directory = os.path.join('./imagenet_c', child)
        if os.path.isdir(directory):
            data_path = os.path.join(directory, '5')
            print(data_path)

# import torch
# import models_mae_shared
# model = models_mae_shared.__dict__['mae_vit_large_patch16'](num_classes=1000, head_type='vit_head', norm_pix_loss='store_true', 
#                                                 #    classifier_depth=classifier_depth, classifier_embed_dim=classifier_embed_dim, 
#                                                 #    classifier_num_heads=classifier_num_heads,
#                                                    rotation_prediction=False)
# model_checkpoint = torch.load('./output/online/model-final.pth', map_location='cpu')
# print(model_checkpoint['bn.running_mean'])
# model.load_state_dict(model_checkpoint)

# print(model)