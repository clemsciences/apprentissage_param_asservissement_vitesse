# -*- coding: Utf-8 -*-
__author__ = 'Clément'

"""
Cet algorithme permet de retrouver les paramètres décrivant au mieux la courbe de vitesse
"""
import random
import time
import numpy as np
import math
#import matplotlib.pyplot as plt
import sys #pour avoir le nombre maximal autorisé

class LisseurGenetique:
    def __init__(self, mesures, duree, nPop, nAperiodique, generation, taux):
        self.MESURES = np.array(mesures)
        self.DUREE = duree
        self.population = []
        self.nPop = nPop
        self.nAperiodique = nAperiodique
        self.generation = generation
        self.taux = taux
        self.T = np.linspace(0, duree, len(mesures)) #discrétisation du temps
        self.__genererPopulation()
        self.__evoluer()


    def __creer_liste(self, taille):
        """
        Retourne une liste de taille taille
        :param taille:
        :return l
        """
        l = []
        for i in range(taille):
            l.append(None)
        return l

    def __genererPopulation(self):
        """
        :param nPop:
        :return: popu  #la population
        """
        self.population = [] #liste des individus
        self.nPseudoPeriodique = self.nPop - self.nAperiodique
        for i in range(self.nPseudoPeriodique):
            self.population.append([])
            self.population[i].append(8524)#random.choice(range(0, 100))) # c -0
            self.population[i].append(30)#random.choice(range(0, len(self.T)))) # mu - 1
            self.population[i].append(math.pow(10, random.random()*1-1)) # beta - 2
            self.population[i].append((-1)**random.choice([1,0])*math.pow(10, random.random()*4)) #omega - 3
            self.population[i].append((-1)**random.choice([1,0])*math.pow(10, random.random()*4))# A - 4
            self.population[i].append(0)#(-1)**random.choice([1,0])*math.pow(10, random.random()*4))# B - 5
            self.population[i].append(random.choice(range(10, 1000))) # eta - 6
            self.population[i].append("P") # "P" pour pseudo-périodique - 7
        for i in range(self.nAperiodique):
            self.population.append([])
            self.population[i].append(random.choice(range(0, 100))) # c - 0
            self.population[i].append(random.choice(range(0, len(self.T)))) # mu - 1
            self.population[i].append(math.pow(10, random.random()*6.5-6)) #beta - 2
            self.population[i].append(math.pow(10, random.random()*6-3)) #alpha - 3
            self.population[i].append(math.pow(10, random.random()*12-6)) #A - 4
            self.population[i].append(math.pow(10, random.random()*12-6)) #B - 5
            self.population[i].append(random.choice(range(10, 1000))) # eta - 6
            self.population[i].append("A") # "A" pour apériodique - 7
    def f_droite(self, c, t):
        return c*t

    def f_aperiodique(self, beta, alpha, A, B, eta, t):
        return math.exp(-beta*t)*(A*math.exp(alpha*t)+B*math.exp(-alpha*t))+eta

    def f_pseudo_periodique(self, beta, omega, A, B, eta, t):
        return  math.exp(-beta*t)*(A*math.cos(omega*t)+B*math.sin(omega*t))+eta

    def __evaluer_droite(self, individu):
        """
        :param individu: c'est lui qu'on évalue
        :return ecart: c'est l'écart par rapport à l'équilibre
        """
        #Faudrait-il créer une classe individu?

        c = individu[0]
        mu = individu[1]
        #print "mu",mu
        valeurs_modele_droite = np.array([self.f_droite(c, t) for t in self.T[:mu]])
        #print self.MESURES[:mu].shape, valeurs_modele_droite.shape
        diff = self.MESURES[:mu] - valeurs_modele_droite
        ecart_droite  = np.dot(diff,diff.T)
        #print ecart_droite
        return ecart_droite


    def __evaluer_courbe(self, individu):
        """
        :param individu: c'est lui qu'on évalue
        :return ecart: c'est l'écart par rapport à l'équilibre
        """
        #Faudrait-il créer une classe individu?
        mu = individu[1]
        beta = individu[2]
        omega = individu[3]
        A = individu[4]
        B = individu[5]
        eta = individu[6]
        print("individu", individu)
        if individu[7] == "P":
            valeurs_modele = np.array([self.f_pseudo_periodique(beta, omega, A, B, eta, t - self.T[mu]) for t in self.T[mu:]])
        elif individu[7] == "A":
            valeurs_modele = np.array([self.f_aperiodique(beta, omega, A, B, eta, t -  self.T[mu]) for t in self.T[mu:]])
        else:
            raise ValueError

        diff = self.MESURES[mu:] - valeurs_modele
        #print self.MESURES[mu:].shape, valeurs_modele.shape
        ecart_courbe  = np.dot(diff,diff.T)
        #print ecart_courbe
        return ecart_courbe

    def __evaluer(self, individu):
        """
        :param individu: c'est lui qu'on évalue
        :return ecart: c'est l'écart par rapport à l'équilibre
        """
        return self.__evaluer_droite(individu) +self.__evaluer_courbe(individu)


    # def __crossover(self, parent1, parent2):
    #     """
    #     On crée les enfants par un simple crossingover si les parents entourent l'équilibre
    #     Si les deux parents sont d'un côté de l'équilibre, alors on garde celui qui est le plus proche de l'équilibre
    #     Cette fonction contient la reproduction, la sélection et le croisement
    #     :param parent1:
    #     :param parent2:
    #     :return:
    #     """
    #     #On duplique le meilleur des parents
    #     ev1 = self.__evaluer(parent1)
    #     ev2 = self.__evaluer(parent2)
    #     if ev1 < ev2:
    #         enfant1 = parent1
    #         enfant2 = []
    #         enfant2.append((3*parent1[0]+parent2[0])/4.)
    #         enfant2.append((3*parent1[1]+parent2[1])/4.)
    #         enfant2.append((3*parent1[2]+parent2[2])/4.)
    #         enfant2.append((3*parent1[3]+parent2[3])/4.)
    #         enfant2.append(parent1[4])
    #     else:
    #         enfant1 = parent2
    #         enfant2 = []
    #         enfant2.append((parent1[0]+3*parent2[0])/4.)
    #         enfant2.append((parent1[1]+3*parent2[1])/4.)
    #         enfant2.append((parent1[2]+3*parent2[2])/4.)
    #         enfant2.append((parent1[3]+3*parent2[3])/4.)
    #         enfant2.append(parent2[4])
    #     return enfant1, enfant2

    def __crossover(self, parent1, parent2):
        """
        On crée les enfants par un simple crossingover si les parents entourent l'équilibre
        Si les deux parents sont d'un côté de l'équilibre, alors on garde celui qui est le plus proche de l'équilibre
        Cette fonction contient la reproduction, la sélection et le croisement
        :param parent1:
        :param parent2:
        :return:
        """
        #On duplique le meilleur des parents
        ev1_droite = self.__evaluer_droite(parent1)
        #print ev1_droite
        ev1_courbe = self.__evaluer_courbe(parent1)
        #print ev1_courbe
        ev2_droite = self.__evaluer_droite(parent2)
        ev2_courbe = self.__evaluer_droite(parent2)
        enfant1 = self.__creer_liste(len(parent1))
        enfant2 = self.__creer_liste(len(parent1))
        if ev1_droite < ev2_droite:
            enfant1[0:2] = parent1[0:2]
            enfant2[0:2] = parent1[0:2]
        else:
            enfant1[0:2] = parent2[0:2]
            enfant2[0:2] = parent2[0:2]

        if ev1_courbe < ev2_courbe:
            enfant1[2:8] = parent1[2:8]
            enfant2[2:8] = parent1[2:8]
        else:
            enfant1[2:8] = parent2[2:8]
            enfant2[2:8] = parent2[2:8]
        return enfant1, enfant2


    def __mutation(self, individu, taux):
        """
        La mutation permet de découvrir d'autres solutions possibles
        :param individu:
        :param taux: permet d'ajuster le nombre de mutation
        :return:
        """
        INDICES = [2,3,4,6]
        if random.random() < taux:
            individu[random.choice(INDICES)] = (-1)**random.choice([1,0])*math.pow(10,random.random()*3-3)
            return individu
        else:
            return individu

    def __evoluer(self):
        #Les individus sont enregistrés dans population
        individus = self.population
        self.meilleurs = []
        minTot = sys.maxint
        for i in range(self.generation):
            population = []
            random.shuffle(individus) #On mélange aléatoirement les individus dans la liste
            for j in range(self.nPop/2):
                #print j
                #La reproduction se fait avec deux individus proches dans la liste individus
                # if individus[2*j] is None:
                #     print "ok",2*j
                # if individus[2*j+1] is None:
                #     print "okk", 2*j+1
                a, b = self.__crossover(individus[2*j], individus[2*j+1])

                #print a,b, "parent enfant"
                population.append(self.__mutation(a, 0.2))
                population.append(self.__mutation(b, 0.2))
            #print population
            individus = population
            
            #On trie pour garder le résultat le plus intéressant
            mini = self.__evaluer(population[0])
            chemin_choisi = 0
            for i in range(1, nPop):
                if mini > self.__evaluer(population[i]):
                    chemin_choisi = i
                    mini = self.__evaluer(population[i])
                    self.meilleurs.append(population[i])
            #min est le chemin qui a la plus petite évaluation
            #A chaque fois qu'on a un meilleur résultat, on l'affiche
            if minTot > mini:
                minTot = mini
                print(population[chemin_choisi], minTot)
            if mini == 0:
                break
            
        self.population = population

    def meilleurs_resultats(self, n):
        for i in range(n):
            mini = i
            #Achtung, hier muss i+1 weniger als n-1 sein !
            for j in range(i+1,self.nPop):
                if self.population[j] < self.population[mini]:
                    mini = j
            if i != mini:
                aux = self.population[i]
                self.population[i] = self.population[mini]
                self.population[mini] = aux
        self.meilleurs = self.population[:n]
        return self.population[:n]


if __name__ == "__main__":
    nom_fichier = "donnees_vitesse.txt"

    # -----------------------------------
    # mesures
    mesures = np.genfromtxt(nom_fichier, delimiter = "\t")
    mesures_vitesse = mesures[:,1]
    mesures_commande = mesures[:,2]
    mesures_reponse = mesures[:,3]
    # durée en ms
    duree = 1.500
    # nombre d'individus dans la population
    nPop = 10000
    # nombre de fonction apériodique
    nAperiodique = 0
    #nombre de génération
    nGeneration = 10
    # taux de mutation
    taux = 0.05
    #nombre de meilleur résultat voulu
    nbResultat = 5
    # ----------------------------------
    t1 = time.time()

    lisseur = LisseurGenetique(mesures_vitesse, duree, nPop, nAperiodique, nGeneration, taux)
    t2 = time.time()
    res =  lisseur.meilleurs_resultats(nbResultat)
    print("res",res)
    t3 = time.time()
    print("en tout : "+str(t3 - t1))
    print("génération : "+str(t2 - t1))
    print("tri : "+str(t3 - t2))
    #C'est ici qu'on récupère la valeur à donner
    T= np.linspace(0, 1.5, len(mesures_vitesse))
    for i in range(nbResultat):
        c = res[i][0]
        mu = res[i][1]
        beta = res[i][2]
        omega = res[i][3]
        A = res[i][4]
        B = res[i][5]
        eta = res[i][6]
        print(c, mu, beta, omega, A, B, eta)
        valeur_modele_droite = np.array([lisseur.f_droite(c, t) for t in T[:mu]])
        valeur_modele_pseudo_periodique = np.array([lisseur.f_pseudo_periodique(beta, omega, A, B, eta, t) for t in T[mu:]])
        valeur_modele = np.concatenate((valeur_modele_droite, valeur_modele_pseudo_periodique))
        # plt.plot(T, valeur_modele, 'r')
        # plt.plot(T, mesures_vitesse, "b")
        # plt.plot(T, mesures_commande, "y")
        # plt.plot(T, mesures_reponse, "g")
        # plt.autoscale()
        # plt.show()
        # plt.cla()
    for i in lisseur.meilleurs:
        c = i[0]
        mu = i[1]
        beta = res[i][2]
        omega = res[i][3]
        A = res[i][4]
        B = res[i][5]
        eta = res[i][6]
        print(c, mu, beta, omega, A, B, eta)
        valeur_modele_droite = np.array([lisseur.f_droite(c, t) for t in T[:mu]])
        valeur_modele_pseudo_periodique = np.array([lisseur.f_pseudo_periodique(beta, omega, A, B, eta, t) for t in T[mu:]])
        valeur_modele = np.concatenate((valeur_modele_droite, valeur_modele_pseudo_periodique))
        # plt.plot(T, valeur_modele, 'r')
        # plt.plot(T, mesures_vitesse, "b")
        # plt.plot(T, mesures_commande, "y")
        # plt.plot(T, mesures_reponse, "g")
        # plt.autoscale()
        # plt.show()
        # plt.cla()

