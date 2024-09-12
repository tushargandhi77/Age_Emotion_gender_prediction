from utils import DL_model

folder_1 = 'UTKFace'
folder_2 = 'emotion/train'

dl = DL_model(folder_1,folder_2)

dl.compile_Fit_And_save(modal='ResNet',gender_batch=150,emotion_batch=32)

