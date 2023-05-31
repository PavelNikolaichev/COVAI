from django.shortcuts import render
from django.views.generic import TemplateView

from django.core.files.storage import FileSystemStorage

import numpy as np
from keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt

from django.conf import settings

IMG_SIZE = 256


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

                file_name = fs.save(file.name, file)
                file_url = fs.url(file_name)
                context['file'] = file_url

                img = load_img(f'./media/{file_name}', target_size=(IMG_SIZE, IMG_SIZE), color_mode='grayscale')
                x = img_to_array(img)
                predicted = getattr(settings, 'MODEL', None).predict(x.squeeze().reshape(1, IMG_SIZE, IMG_SIZE, 1))
                plt.imsave(f'./media/{file_name}_output.jpg', (predicted.squeeze() > 0.3).astype(np.uint8),
                           cmap='Greys')
                context['predicted'] = f'/media/{file_name}_output.jpg'

        return render(request, 'output.html', context)