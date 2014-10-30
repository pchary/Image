import numpy as np                      # Pour manipuler l'image = tableau 2D de nombres
import matplotlib.pyplot as plt         # Pour les tr
from scipy import misc                  # Pour charger l'image de lena :)
#import urllib, cStringIO
#from PIL import Image
import requests
from io import BytesIO

def TF(img):
    """ Calcul de la transforme de Fourier de l'image img"""
    
    spectre = np.fft.fft2(img)          # TF
    sp_shifte = np.fft.fftshift(spectre)# Il faut dcaler le spectre, sinon les basses frquences sont dans les coins et non au centre...
    return sp_shifte

def filtrage(spectre,type="passe haut",taille=40):
    """ Application d'un filtre passe haut ou passe bas, de forme carre et de dimension taille, au spectre de Fourier 2D de l'image"""

    nblig, nbcol = spectre.shape            # nombre de lignes et de colonnes de l'image
    clig,ccol = nblig//2 , nbcol//2         # Coordonns du centre
    
    if type == "passe haut":
        sp_filtre = spectre.copy() 
        sp_filtre[clig-taille:clig+taille, ccol-taille:ccol+taille] = 0 # On met  0 le carr central de sp_shifte
    elif type == "passe bas":
        masque = np.zeros((nblig,nbcol))    # On utilise un masque pour obtenir l'action complmentaire du cas prcdent
        masque[clig-taille:clig+taille, ccol-taille:ccol+taille] = 1
        sp_filtre = spectre*masque
    else:
        sp_filtre = spectre
    return sp_filtre

def TF_inv(sp_filtre):
    """ Calcul de la transforme de Fourier inverse du spectre filtr """ 

    sp_filtre_shifte = np.fft.ifftshift(sp_filtre)  # On redcale le spectre pour remettre les basses frquences dans les coins afin que la TF inverse fonctionne correctement.
    img_filtree = np.fft.ifft2(sp_filtre_shifte)    # TF inverse
    return np.abs(img_filtree)

def main(type,taille):
    """ Programme principal """
    
    #img = misc.lena()
    #img = plt.imread(r"E:\Prepa\Informatique\Python\Moi\2eme annee\Projets\Traitement Image\moonlanding.png") # Exemple de chargement d'une image avec chemin complet sur le disque (ou utiliser os)
    url = "https://raw.githubusercontent.com/scipy-lectures/scipy-lecture-notes/master/data/moonlanding.png"
    response = requests.get(url)
    img = plt.imread(BytesIO(response.content))
    #image ici: https://github.com/scipy-lectures/scipy-lecture-notes/blob/master/data/moonlanding.png
    
    sp_shifte = TF(img)
    sp_filtre = filtrage(sp_shifte,type,taille)
    img_filtree = TF_inv(sp_filtre)

    sp_shifte_module = 20*np.log(1+np.abs(sp_shifte))   # Echelle logarithmique pour bien voir toutes les composantes spectrales d'amplitudes trs diffrentes, le +1 est pour viter les valeurs nulles dans le log.
    sp_filtre_module = 20*np.log(1+np.abs(sp_filtre)) 
    
    # Tracs
    plt.figure()
    
    plt.subplot(221),plt.imshow(img, cmap = 'gray')
    plt.title('Image de depart'), plt.xticks([]), plt.yticks([])
    plt.subplot(222),plt.imshow(sp_shifte_module, cmap = 'gray')
    plt.title('Spectre (TF)'), plt.xticks([]), plt.yticks([])
    plt.subplot(223),plt.imshow(img_filtree, cmap = 'gray')
    plt.title('Image filtree '+type), plt.xticks([]), plt.yticks([])
    plt.subplot(224),plt.imshow(sp_filtre_module, cmap = 'gray') 
    plt.title('Spectre filtre '+type), plt.xticks([]), plt.yticks([])
    
    plt.show()
    
if __name__ == '__main__':
    main("passe haut",40)
    main("passe bas",40)


