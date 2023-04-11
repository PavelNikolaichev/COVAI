from django.shortcuts import render
from django.views.generic import TemplateView

from django.core.files.storage import FileSystemStorage

import numpy as np
from keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt
import tensorflow as tf

from .NNmodels import *

IMG_SIZE = 256

model = UEfficientNet(input_shape=(IMG_SIZE, IMG_SIZE, 3), dropout_rate=0.5)
model.load_weights('./models/checkpointUnetPlusPlus')


class BinaryMeanIoU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        th = 0.5
        return super().update_state(tf.cast(y_true > th, tf.int32), tf.cast(y_pred > th, tf.int32), sample_weight)


class IndexView(TemplateView):
    template_name = 'index.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = 'Проверить снимок на Covid'
        return context

    def predictImage(request):
        context = {
            'title': 'Проверить снимок на Covid'
        }

        if request.POST:
            if request.FILES:
                file = request.FILES['file']
                fs = FileSystemStorage()

                fileName = fs.save(file.name, file)
                fileUrl = fs.url(fileName)
                context['file'] = fileUrl

                img = load_img(f'./media/{fileName}', target_size=(IMG_SIZE, IMG_SIZE), color_mode='grayscale')
                x = img_to_array(img)
                print(x.squeeze().shape)
                predicted = model.predict(x.squeeze().reshape(1, IMG_SIZE, IMG_SIZE, 1))
                print(predicted.shape)
                print((predicted.squeeze() > 0.3).sum())
                plt.imsave(f'./media/{fileName}_output.jpg', (predicted.squeeze() > 0.3).astype(np.uint8),
                           cmap='Greys')
                context['predicted'] = f'/media/{fileName}_output.jpg'

        return render(request, 'output.html', context)
