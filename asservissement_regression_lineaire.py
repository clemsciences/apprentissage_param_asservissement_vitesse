# -*- coding: Utf-8 -*-
__author__ = 'Clément'


import numpy as np
import time, math

class AsservissementRegression:
    def __init__(self, y_mesurees, duree):
        self.MESURES = y_mesurees
        self.T = np.linspace(0, duree, len(y_mesurees))
        print "taille", len(y_mesurees)
        print "durée",duree, len(self.T)
    def calculer_c(self, mu):
        #print np.dot(self.T[:mu].T, self.T[:mu])
        return np.dot(np.dot(1/np.dot(self.T[:mu].T, self.T[:mu]), self.T[:mu].T), self.MESURES[:mu])

    def calculer_erreur(self):
        """
        :param individu: c'est lui qu'on évalue
        :return ecart: c'est l'écart par rapport à l'équilibre
        """
        ecart_droite = []
        #Faudrait-il créer une classe individu?
        mu = np.argmax(self.MESURES)-1
        print "mu",mu
        c = self.calculer_c(mu)
        print "c", c
        """
        mu 30
        c 8524.0818235
        erreur 320905.949737
        taille 229
        durée 1.5 229
        mu 31
        c 8522.93178213
        erreur 320911.094659
        mu 32
        c 8496.58986175
        erreur 323988.940092
        mu 33
        c 8450.34965035
        erreur 334758.041958
        """
        valeurs_modele_droite = np.array([f_droite(c, t) for t in self.T[:mu]])
        #print self.MESURES[:mu].shape, valeurs_modele_droite.shape
        diff = self.MESURES[:mu] - valeurs_modele_droite
        ecart = np.dot(diff,diff.T)
        return ecart

def f_aperiodique(beta, alpha, A, B, t):
    return math.exp(-beta*t)*(A*math.exp(alpha*t)+B*math.exp(-alpha*t))
def f_pseudo_periodique(beta, omega, A, B, t):
    """
    print "f_pseudo_perio"
    print beta, omega, A, B, t
    print math.exp(-t*beta)
    print math.cos(omega*t)
    print math.sin(omega*t)
    """
    return  math.exp(-beta*t)*(A*math.cos(omega*t)+B*math.sin(omega*t))
def f_droite(c, t):
    return c*t



if __name__ == "__main__":
    nom_fichier = "donnees_vitesse.txt"

    #-----------------------------------
    #mesures
    mesures = np.genfromtxt(nom_fichier, delimiter = "\t")
    mesures_vitesse = mesures[:,1]
    mesures_commande = mesures[:,2]
    mesures_reponse = mesures[:,3]
    #durée en ms
    duree = 1.500
    #----------------------------------
    t1 = time.time()
    #C'est ici qu'on récupère la valeur à donner
    asserivssement = AsservissementRegression(mesures_vitesse, duree)
    erreur = asserivssement.calculer_erreur()
    print "erreur", erreur
    """
    plt.plot(T, valeur_modele, 'r')
    plt.plot(T, mesures_vitesse, "b")
    plt.plot(T, mesures_commande, "y")
    plt.plot(T, mesures_reponse, "g")
    plt.autoscale()
    plt.show()
    plt.cla()
    """