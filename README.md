# COVID-19 Segmenter
An app to detect COVID-19 on CT-scans and segment infected zones.
This app is written on django and keras.

Also, it should be noted that this code was written for a hackathon, so the code was made
as fast as I could, without refactoring it and beautiful architecture. Therefore, I want to
refactor it one day.

# TODO
1. Make model part work like a REST-service.
2. Improve this code, especially in controller part.

# Installation and running
1. ```pip install -r requirements.txt```
2. ```python manage.py migrate```
3. ```python manage.py runserver```