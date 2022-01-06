"""
Name: Alek Michael
A-J Protoypes
"""

import numpy as np
import prototypes 
from sklearn.model_selection import train_test_split

# Num of Samples
n_samples = 100
#Chance to invert bit
noise = .1

def contains(A, B) :
    for arr in A :
        if np.array_equal(arr, B) :
            return True
    return False

def generateNoise(proto) :
    variants = [ proto ]
    counter = n_samples - 1
    while counter > 0 :
        noisy = proto.copy()
        mask = np.random.binomial(1, noise, noisy.shape).astype(bool)
        
        for i in range(len(noisy)) :
            if mask[i] == True :
                if noisy[i] == 1 :
                    noisy[i] -= 1
                else :
                    noisy[i] += 1
        if contains(variants, noisy) is False :
            variants.append(noisy)
            counter -= 1
    return np.array(variants)

# Generate Noise Variants using genereateNoise 
def generateNoiseVariants() :
    return list(map(generateNoise, (prototypes.A, prototypes.B, prototypes.C, prototypes.D, prototypes.E, prototypes.F, prototypes.G, prototypes.H, prototypes.I, prototypes.J)))
     
def splitVariants() :
    variants = generateNoiseVariants()
    allVariants = []
    variantAnswer = []
    answers = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    answerIndex = -1
    for variant in variants :
        answerIndex += 1
        for v in variant :
            allVariants.append(v)
            #variantAnswer.append(answers[answerIndex])
            variantAnswer.append(ord(answers[answerIndex]))
    
    return train_test_split(allVariants, variantAnswer, train_size=0.6, test_size=0.4)
