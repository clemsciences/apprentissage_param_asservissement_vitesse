# -*- coding: Utf-8 -*-
__author__ = 'Clément'


import numpy as np
import math
# import matplotlib.pyplot as plt
import time
import sys


class RecuitSimule:
    def __init__(self, y_mesurees, duree):
        self.y_mesurees = y_mesurees
        self.T = np.linspace(0, duree, len(y_mesurees))

    def ksi(self, beta, omega, A, B):
        """
        :param beta:
        :param A:
        :param B:
        :param omega:
        :param y_mesuree:
        :return:
        """
        # calcul des y selon le modèle deviné
        y_modele = np.array([f_pseudo_periodique(beta, omega, A, B, t) for t in self.T])
        return (1/2.)*np.dot((self.y_mesurees - y_modele), (self.y_mesurees - y_modele).T)

    def gradient_ksi(self, beta, omega, A, B):
        # calcul des y selon le modèle deviné
        y_modele = np.array([f_pseudo_periodique(beta, omega, A, B, t) for t in self.T])
        gamma = (y_modele - self.y_mesurees)
        der_beta =  np.array([-1*math.exp(-t*beta)*(B*math.cos(omega*t)-A*math.sin(omega*t)) for t in self.T])
        der_A = np.array([math.exp(-t*beta)*math.cos(omega*t) for t in self.T])
        der_B = np.array([math.exp(-t*beta)*math.sin(omega*t) for t in self.T])
        der_omega = np.array([math.exp(-t*beta)*(B*math.cos(omega*t)-A*math.sin(omega*t)) for t in self.T])
        return np.array([np.dot(gamma, der_beta.T), np.dot(gamma, der_omega.T), np.dot(gamma,der_A.T), np.dot(gamma, der_B.T)])
        # return np.array([np.dot(der_beta,der_beta.T)- np.dot(self.y_mesurees, der_beta.T), np.dot(der_A,der_A.T) - np.dot(self.y_mesurees, der_A.T), np.dot(der_B,der_B.T) - np.dot(self.y_mesurees, der_B.T), np.dot(der_omega,der_omega.T) - np.dot(self.y_mesurees, der_omega.T)])
        # return np.array([np.dot(der_beta,der_beta.T)- np.dot(self.y_mesurees, der_beta.T), np.dot(der_A,der_A.T) - np.dot(self.y_mesurees, der_A.T), np.dot(der_B,der_B.T) - np.dot(self.y_mesurees, der_B.T), np.dot(der_omega,der_omega.T) - np.dot(self.y_mesurees, der_omega.T)])

    def algo_descente_gradient(self, initialisation, nb_iter):
        beta, omega, A, B = initialisation
        # alpha = np.array([0.001,0.1,0.1,0.1])
        alpha = 0.1
        eval_dernier = sys.maxint
        for i in range(nb_iter):
            grad = self.gradient_ksi(beta, omega, A, B)
            # print grad
            # beta, omega, A, B = np.array([beta, omega, A, B]) - np.dot(alpha, grad.T)
            beta, omega, A, B = np.array([beta, omega, A, B]) - alpha * grad
            print(beta, omega, A, B)
            valeur = self.ksi(beta, omega, A, B)

            if eval_dernier > valeur:
                alpha *= 0.1
            else:
                alpha *= 10

            # print(valeur)
            eval_dernier = valeur
        return beta, omega, A, B


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


if __name__ == "__main__":
    nom_fichier = "donnees_vitesse.txt"

    # -----------------------------------
    #mesures
    mesures = np.genfromtxt(nom_fichier, delimiter = "\t")
    mesures_vitesse = mesures[:,1]
    mesures_commande = mesures[:,2]
    mesures_reponse = mesures[:,3]
    # durée en ms
    duree = 1.500
    # paramètres initiaux
    initialisation = (1, 1, 1, 1)
    # nombre d'itérations de l'algorithme de descente de gradient
    nb_iter = 30
    # ----------------------------------
    t1 = time.time()
    # C'est ici qu'on récupère la valeur à donner
    modeleur = RecuitSimule(mesures_vitesse, duree)
    beta, omega, A, B = modeleur.algo_descente_gradient(initialisation, nb_iter)
    T = np.linspace(0, duree, len(mesures_vitesse))
    valeur_modele = np.array([f_pseudo_periodique(beta, omega, A, B, t) for t in T])
    """
    plt.plot(T, valeur_modele, 'r')
    plt.plot(T, mesures_vitesse, "b")
    plt.plot(T, mesures_commande, "y")
    plt.plot(T, mesures_reponse, "g")
    plt.autoscale()
    plt.show()
    plt.cla()
    """