from main import *
import numpy as np

track_id = int(input("Enter track id:\n"))
acousticness = float(input("Enter acousticness:\n"))
danceability = float(input("Enter danceability:\n"))
energy = float(input("Enter energy:\n"))
instrumentalness = float(input("Enter instrumentalness:\n"))
liveness = float(input("Enter liveness:\n"))
speechiness = float(input("Enter speechiness:\n"))
tempo = float(input("Enter tempo:\n"))
valence = float(input("Enter valence:\n"))

input = np.array([[track_id,acousticness,danceability,energy,instrumentalness,liveness,speechiness,tempo,valence]])

model = final_algorithm()
res = model.predict(input)

if res[0] == 1:
    print("Rock")
else:
    print("Hip-Hop")