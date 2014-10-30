# Pour manipuler l'image = tableau 2D de nombres
import numpy as np
# Pour les traces
import matplotlib.pyplot as plt
# Pour charger l'image de lena :)
import requests
from io import BytesIO


def TF(img):
    """ Calcul de la transforme de Fourier de l'image img"""
    spectre = np.fft.fft2(img)            # TF
    # Il faut decaler le spectre, sinon les basses frequences
    # sont dans les coins et non au centre...
    sp_shifte = np.fft.fftshift(spectre)
    return sp_shifte


def filtrage(spectre, type="passe haut", proportion=0.03):
    """ Application d'un filtre passe haut ou passe bas,
    de forme carree et de dimension taille, au spectre de Fourier
    2D de l'image"""
    # nombre de lignes et de colonnes de l'image
    nblig, nbcol = spectre.shape
    # Coordonnees du centre
    clig, ccol = nblig // 2, nbcol // 2
    print nblig, nbcol
    masque = np.zeros((nblig, nbcol))
    taille = int(proportion * (nblig + nbcol) / 2.0)
    print "taille", taille
    masque[clig-taille:clig+taille, ccol-taille:ccol+taille] = 1
    if type == "passe haut":
        masque = 1 - masque
    sp_filtre = spectre * masque
    return sp_filtre


def TF_inv(sp_filtre):
    """ Calcul de la transforme de Fourier inverse du spectre filtr """
    # On redecale le spectre pour remettre les basses frequences
    # dans les coins afin que la TF inverse fonctionne correctement.
    sp_filtre_shifte = np.fft.ifftshift(sp_filtre)
    # TF inverse
    img_filtree = np.fft.ifft2(sp_filtre_shifte)
    return np.abs(img_filtree)


def execute(url, ltype, proportion=0.03):
    """ Programme principal """
    response = requests.get(url)
    img = plt.imread(BytesIO(response.content))
    sp_shifte = TF(img)
    sp_filtre = filtrage(sp_shifte, ltype, proportion)
    img_filtree = TF_inv(sp_filtre)
    # Echelle logarithmique pour bien voir toutes les composantes spectrales
    # d'amplitudes trs diffrentes, le +1 est pour viter les valeurs nulles
    # dans le log.
    sp_shifte_module = 20*np.log(1+np.abs(sp_shifte))
    sp_filtre_module = 20*np.log(1+np.abs(sp_filtre))

    # Traces
    plt.figure()
    plt.subplot(221), plt.imshow(img, cmap='gray')
    plt.title('Image de depart'), plt.xticks([]), plt.yticks([])
    plt.subplot(222), plt.imshow(sp_shifte_module, cmap='gray')
    plt.title('Spectre (TF)'), plt.xticks([]), plt.yticks([])
    plt.subplot(223), plt.imshow(img_filtree, cmap='gray')
    plt.title('Image filtree ' + ltype), plt.xticks([]), plt.yticks([])
    plt.subplot(224), plt.imshow(sp_filtre_module, cmap='gray')
    plt.title('Spectre filtre ' + ltype), plt.xticks([]), plt.yticks([])
    plt.show()

if __name__ == '__main__':
    url_root = "https://raw.githubusercontent.com/dombrno/Image/master/SampleImages/"
    url_names = ["Nebuleuse", "peter", "NotreDame", "Panthera_tigris_tigris",
                 "Passat", "Torre_Guinigi", "shark"]
    url_list = [url_root + url_name + ".png" for url_name in url_names]
    for url in url_list:
        execute(url, "passe haut", 0.025)
        execute(url, "passe bas", 0.025)
