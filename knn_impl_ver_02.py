import pandas as pd
from random import randrange
from csv import reader
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns
import random

"""
Επιστρέφει μια λίστα με λίστες που κάθε λίστα ξεχωριστά αντιπροσωπεύει τα χαρακτηριστικά του όγκου ενώς ασθενή.
Αρχικά δημιουργείται μια κενή λίστα με όνομα dataset. Στην συνέχεια διαβάζουμε τα δεδομένα από κάθε σειρά 
του csv file που έχουμε προσδιορίσει με το path που βρίσκεται στο filename και κάνουμε κάθε φορά
append την σειρά που διαβάζουμε στην λίστα dataset. Τέλος η λίστα dataset, η οποία περιέχει τα δεδομένα
κάθε σειράς του csv που διαβάσαμε προηγουμένως, σε μορφή λίστας επίσης, επιστρέφεται.
"""
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

"""
Κάνουμε όλες τις τιμές του datase float. Μέσα στο for loop παίρνουμε κάθε στήλη του dataset
και μετατρέπουμε την τιμή που βρίσκεται εκεί από str σε float.
"""
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())
        


"""
Η συνάρτηση cross_validation_split παίρνει σαν είσοδο το dataset και τον αριθμό των folds (n_folds).
Δημιουργεί την κενή λίστα datset_split όπου θα μπούν τα folds που θα έχουμε φτιάξει από το dataset.
Αντιγράφει το dataset στην dataset_copy. Βρίσκει το μέγεθος κάθε fold (fold_size). Στην περίπτωση μας
κάθε fold θα αποτελείται από 113 εγγραφές αφού χρησιμοποιούμε n_folds = 5.
Στο for loop το οποίο τρέχει για 5 φορές βάζουμε μέσα στην λίστα fold 113 τυχαίες εγγραφές, κάθε fold
εκχωρείται στο dataset_split. Τέλος τι dataset_split επιστρέφεται.
"""
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for _ in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split 


"""
Η συνάρτηση cross_validation_split_without_randrage είναι ίδια με την παραπάνω με την μόνη διαφορά
πως δεν βάζοουμε τυχαίες εγγραφές από το dataset_copy κάθε φορά μέσα στο fold με την χρήση του randrange, 
αλλά με την σειρά, αρχίζοντας από την εγγραφή 0. Το k-fold cross validation χρησιμεύει στο να προσδιορίσουμε το 
πόσο καλά το μοντέλο μας αποδίδει σε καινούργια δεδομένα. Σηνύθως επιλέγεται ένας μονός αριθμός για πλήθος 
δεδομένων με μερικές εκατοντάδες εγγραφές.
"""
def cross_validation_split_without_randrage(dataset, n_folds): 
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    index = 0
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            fold.append(dataset_copy[index])
            index += 1
        dataset_split.append(fold)
    return dataset_split
   

            
"""
Συγκρίνει την πραγματική τιμή του target variable με αυτή που προβλέφθηκε και 
αυξάνει τον μετρητή correct κατά 1 αν είναι ίσες. Τέλος επιστρέφει το ποσοστό επί της 100 των σωστών προβλέψεων
"""
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0
 
"""
Στην evaluate_algorithm παίρνουμε σαν παραμέτρους το dataset που περίεχει όλες τις λίστες με τα χαρακτηριστικά των όγκων των ασθενών,
το algorithm που θα χρησιμοποιηθεί για τις προβλέψεις (knn), το n_folds που είναι ο αριθμός των folds και το *args που είναι ο αριθμός των γειτόνων.
Καλούμε την συνάρτηση cross_validation_split και παίρνουμε πίσων μια λίστα με 5 λίστες (όσα και τα folds) που κάθε τέτοια λίστα περιέχει τις λίστες
που κάθε μια περιέχει τα χαρακτηριστικά του όγκου του ασθενή. Αρχικοποιούμε την κενή λίστα scores.
Μέσα στο for loop για αριθμό ίσον με αυτόν των folds, κάθε φορά,δημιουργούμε την λίστα train_set στην οποία εκχωρούμε την λίστα folds. 
Αφαιρούμε από αυτή την λίστα ένα από τα folds(αυτό που υποδικνύεται από τα fold) και κάνουμε το train_set μια ενιαία λίστα που περιέχει τις εγγραφές των ασθενών (Επίσης λίστες με χαρακτηριστικά των όγκων).
Δημιουργούμε την κενή λίστα test_set. Στο εσωτερικό for loop για κάθε γραμμή από το fold που έχει αφαιρεθεί προηγουμένως κάνουμε την target variable ίση με None (row_copy[0] = None).
Καλούμε την συνάρτηση k_nearest_neighbors (algorithm) και παίρνουμε μια λίστα με 0 και 1 (προβλέψεις για καλοήθεις, κακοήθεις). Στην συνέχεια καλούμε 
την συνάρτηση accuracy_metric από την οποία παίρνουμε το accuracy% για το fold που επιλέγεται σε κάθε loop και το κάνουμε append στην λίστα scores.
Τέλος επιστρέφουμε την scores.     
"""
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split_without_randrage(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        #print(train_set[0],"\n")
        train_set.remove(fold)
        train_set = sum(train_set, []) # Κάνει τα εναπομείναντα folds μια ενιαία λίστα που περιέχει τις εγγραφές των ασθενών (Επίσης λίστες με χαρακτηριστικά των όγκων).
        #print(train_set[0],"\n")
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[0] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[0] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores



 

"""
Στην συνάρτηση euclidean_distance σαν παραμέτρους παίρνουμε το row1 (test_row) και το row2 (train_row).
Μέσα στο for loop υπολογίζουμε το τετράγωνο της διαφοράς των τιμών για κάθε εγγραφή από την δεύτερη τιμή και μετά,
γιατί η πρώτη τιμή στο row2 είναι το target variable και στο row1 είναι None. Τέλος επιστρέφουμε την ρίζα της απόστασης.
"""
def euclidean_distance(row1, row2):
    distance = 0.0  
    for i in range(len(row1)-1):
        distance += (row1[i + 1] - row2[i + 1])**2
    return sqrt(distance)
 
"""
Στην συνάρτηση get_neighbors παίρνουμε σαν παραμέτρους την λίστα train που περιέχει τις εγγραφές που γνωρίζουμε την κλάση τους, το test row,
που είναι μια γραμμή-εγγραφή από την λίστα test στην οποία δεν γνωρίζουμε το target variable και θέλουμε να το προβλέψουμε και το num_neighbors που
είναι ο αριθμός των γειτόνων. 
Αρχικά δημιουργούμε μια κενή λίστα με όνομα distances. Μέσα στο for loop παίρνουμε κάθε φορά μια σειρά train_row από την λίστα train.
Στην μεταλητή dist εκχωρούμε την απόσταση των δύο λιστών-εγγραφών που έχει υπολογιστεί από την συνάρτηση euclidean_distance. Έπειτα εκχωρούμε
κάθε φορά την απόσταση που έχουμε βρεί μαζί με το train_row από το οποίο το test_row απέχει την συγκεκριμένη απόσταση, σε μορφή tuple (train_row, απόσταση),
στην λίστα distances. Ταξινομούμε σε αύξουσα σειρά την λίστα με τα tuples ως προς την απόσταση. Δημιουργούμε την κενή λίστα neighbors. 
Μέσα στο for loop και για αριθμό loops ίσο με αυτό του num_neighbors εκχωρούμε στην λίστα neighbors τις λίστες που παίρνουμε από τα tuples με τα 
χαρακτηριστικά των neighbors που απέχουν την μικρότερη απόσταση. Τέλος επιστρέφουμε την λίστα neighbors.
"""
def get_neighbors(train, test_row, num_neighbors):
	distances = list()
	for train_row in train:
		dist = euclidean_distance(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return neighbors 


"""
Η συνάρτηση predict_classification καλείται απο την k_nearest_neighbors για κάθε γραμμή από την λίστα test. Παίρνει σαν παραμέτρους το train,
που είναι μια λίστα που περιέχει λίστες με τις εγγραφές των ασθενών για τα χαρακτηριστικά των όγκων που έχει παρουσιάσει ο καθέναςμ το test_row,
που είναι μια γραμμή από την λίστα test (η λίστα test θυμίζουμε περιέχει λίστες-εγγραφές) και το num_neighbors που είναι ο αριθμός των γειτόνων.
Έπειτα καλούμε την συνάρτηση get_neighbors από την οποία επιστρέφεται μια λίστα που περιέχει τις λίστες-εγγραφές των κοντινότερων γειτόνων,
Στην λίστα output_values εκχωρούμε το target_variable (καλοήθεις(0), κακοήθεις(1)) των γειτόνων. Τέλος στο prediction εκχωρούμε την τιμή 
που εμφανίζεται περισσότερες φορές στην λίστα output_values και το επιστρέφουμε.
"""
def predict_classification(train, test_row, num_neighbors):
	neighbors = get_neighbors(train, test_row, num_neighbors)
	output_values = [row[0] for row in neighbors]
	prediction = max(set(output_values), key=output_values.count)
	return prediction

    
"""
Στην συνάρτηση k_nearest_neighbors παίρνουμε σαν παραμέτρους το train που είναι η λίστα με τις εγγραφές των ασθενών που ξέρουμε το target variable τους (0, 1),
την λίστα test που περιέχει τις εγγραφές των ασθενών στις οποίες δεν γνωρίζουμε σε ποιο class ανήκουν (0, 1) και το num_neighbors που ε΄ίναι ο αριθμός των
γειτόνων.
Αρχικά δημιουργούμε μια κενή λίστα με όνομα predictions. Στην συνέχεια στο for loop παίρνουμε σε κάθε loop μια γραμμή από την λίστα test. Καλούμε την 
συνάρτηση predict_classification και παίρνουμε στο output 0 ή 1. Το κάνουμε αυτό για όλες τις γραμμές της λίστας test και κάθε φορά κάνουμε append το output
στην λίστα predictions.
Στο if ελέγχουμε αν έχουμε δώσει καινούργια δεδομένα για νέους ασθενείς. Σε αυτή την περίπτωση το target variable θα είναι ίσο με -1 και 
θα μπούμε μέσα στο if και θα τυπώσουμε τις προβλέψεις. Αντίθετα θα μπούμε στο else θα τυπώσουμε το σωστό target variable, τις προβλέψεις και τον 
αριθμό των λάθος προβλέψεων. 

"""
def k_nearest_neighbors(train, test, num_neighbors):
    predictions = list()
    for row in test:
        output = predict_classification(train, row, num_neighbors)
        predictions.append(output)
   
    if test[0][0] == -1:
        for i in range(len(test)):
            print('Predicted: ',predictions[i], '\n' )
    else:         
        wrong_pred = 0
        for i in range(len(test)):
            print('Expected %d, Got %d.' % (test[i][0], predictions[i])) 
            if test[i][0] != predictions[i]:
                wrong_pred += 1
        print('At', len(test), ' cases we get ', wrong_pred, 'wrong predictions\n')

 




"""
Συνάρτηση η οποία χρησιμεύει στο να βρούμε τον βέλτιστο αριθμό 
γειτόνων με τους οποίους πετυχαίνουμε το μέγιστο accuracy.
Η συνάρτηση δέχεται σαν παραμέτρους τους αριθμούς των γειτόνων και το dataset.
Καλεί μέσα στο for loop την συνάρτηση evaluate_algorithm για κάθε τιμή του K.
Ακόμα κάνει append στην λίστα mean_acc το mean accuracy που πετυχαίνεται κάθε φορά με κάθε ξεχωριστή τιμή του Κ
και στην λίστα neighbors αποθηκεύει την τιμή του K.
Τέλος σχεδιάζει την γραφική παράσταση του mean_accuracy ως προς το K.
"""
def find_best_K(K, dataset):

    mean_acc = []
    neighbors = []
    for num in range(K):
          scores = evaluate_algorithm(dataset, k_nearest_neighbors, 7, num + 1)
          print('Scores: %s' % scores)
          print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))  
          mean_acc.append((sum(scores)/float(len(scores))))
          neighbors.append(num + 1)


    plt.plot(neighbors, mean_acc)
    plt.xlabel('Neighbors')
    plt.ylabel('Mean accuracy')
    plt.show() 


"""
Βρίσκουμε το mean accuracy ανάλογα με τον αριθμό των folds και για 5 γείτονες. Δημιουργούμε δύο κενές λίστες με όνομα
mean_acc και folds. Στο for loop και για αριθμό folds ίσο με 2 έως fold_num κάνουμε evaluate την απόδοση του αλγορίθμου μας
με την χρήση 5 γειτόνων και αριθμό folds ίσο με num (2 έως fold_num). Έπειτα τυπώνουμε το ποσοστό επί της 100 του mena accuracy
και το κάνουμε append στην λίστα mean_acc όπως και το num στην λίστα folds. Τέλος σχεδιάζεται το plot bar που 
δείχνει το mean accuracy με βάση τον αριθμό των folds.
"""
def fold_num_and_accuracy(fold_num, dataset):
    mean_acc = []
    folds = []    
    for num in range(2, fold_num):
        scores = evaluate_algorithm(dataset, k_nearest_neighbors, num, 5)   
        #print('Scores: %s' % scores)
        print('Mean Accuracy: %.3f%%'%(sum(scores)/float(len(scores))), ' for {}'.format(num),' folds.\n')  
        mean_acc.append((sum(scores)/float(len(scores))))
        folds.append(num)
       
    plt.bar(folds, mean_acc, color ='maroon',  width = 0.4)



"""
Μετατρέπουμε την στήλη diagnosis τα M -> 1 και τα B -> 0
"""
def diagnosis_value(diagnosis): 
	if diagnosis == 'M': 
		return 1
	else: 
		return 0

"""
Φορτώνουμε τα δεδομένα μας από το csv file και κάνουμε drop,
τις στήλες Unnamed: 32 και την id. Στην συνέχεια καλούμε την συνάρτηση
diagnosis_value και κάνουμε το M -> 1 και το B -> 0 από την στήλη diagnosis.
Στην συνέχεια εξάγουμε τα δεδομένα σε μορφή csv στον φάκελο data_2.csv και 
τέλος κάνουμε open το παραπάνω αρχείο, κάνουμε skip μια γραμμή και γράφουμε
στο αρχείο final_data τα δεδομένα μας.
"""

# Ερώτημα 4 ΕΠΕΞΕΡΓΑΣΙΑ ΔΕΔΟΜΕΝΩΝ.
def process_data():
    df_1 = pd.read_csv('C:/Users/user/Desktop/ERGASIES_&_ARXEIA/Διαχείριση_Γνώσης_2/data.csv')
    df_1.info()
    df_2 = df_1.drop(['Unnamed: 32', 'id'], index=None, axis = 1)
    df_2['diagnosis'] = df_2['diagnosis'].apply(diagnosis_value) 
    df_2.to_csv('C:/Users/user/Desktop/ERGASIES_&_ARXEIA/Διαχείριση_Γνώσης_2/data_2.csv', index=False)
    
    with open('C:/Users/user/Desktop/ERGASIES_&_ARXEIA/Διαχείριση_Γνώσης_2/data_2.csv') as f:
        with open('C:/Users/user/Desktop/ERGASIES_&_ARXEIA/Διαχείριση_Γνώσης_2/final_data.csv','w') as f1:
            next(f) # skip header line
            for line in f:
                f1.write(line)

    dataset = load_csv('C:/Users/user/Desktop/ERGASIES_&_ARXEIA/Διαχείριση_Γνώσης_2/final_data.csv')
    for i in range(len(dataset[0])):
    	str_column_to_float(dataset, i)
    #Κάνουμε την πρώτη στήλη int.
    for i in range(len(dataset)):
        dataset[i][0] = int(dataset[i][0])
    
    return dataset

"""
Συνάρτηση για περαιτέρω μελέτη των δεδομένων που έχουμε επιλέξει. Δημιουργούμε ένα pandas dataframe 
από το αρχείο data_2.csv που είναι ένα csv file που η πρώτη γραμμή περιέχει τα ονόματα των στηλών.
Βρίσκουμε και τυπώνουμε το mean, το median και το std. 
Εμφανίζουμε 4 bar plots με την βιβλιοθήκη seaborn στα οποία μπορούμε να δουμε το distribution ορισμένων δεδομένων του dataset μας.
Εμείς βρίσκουμε το distribution του radius_mean, του perimeter_mean, του area_mean και του texture_mean.
"""
def study_data():
    df_dataset = pd.read_csv('C:/Users/user/Desktop/ERGASIES_&_ARXEIA/Διαχείριση_Γνώσης_2/data_2.csv')
    
    print(df_dataset.mean())
    print(df_dataset.median())
    print(df_dataset.std())
    
    sns.displot(df_dataset, x = df_dataset['radius_mean'], bins = list(range(10, 41)))
    sns.displot(df_dataset, x = df_dataset['perimeter_mean'], bins = list(range(90,200,10)))
    sns.displot(df_dataset, x = df_dataset['area_mean'], bins = list(range(400, 2600, 100)))
    sns.displot(df_dataset, x = df_dataset['texture_mean'], bins = list(range(10, 41)))


"""
Συνάρτηση η οποία παίρνει σαν παραμέτρους το df_dataset που είναι ένα pandas dataframe, το dataset σε μορφή λιστών για κάθε εγγραφή 
και το num_of_patients που είναι ο αριθμός των νέων ασθενών για τους οποίους θα προσθέσουμε καινούργια δεδομένα.
Αρχικά παίρνουμε στην λίστα columns τα ονόματα των στηλών από το df_dataset. Δημιουργούμε δύο κενές λίστες, την new_patient 
και την new_patients_list. Μέσα στο for loop και για αριθμό επαναλήψεων ίσο με num_of_patients εκχωρούμε στην πρώτη 
θέση new_patient[0] το -1. Στο εμφωλευμένο for loop και για αριθμό επαναλήψεων ίσο με 30 βάζουμε τυχαίες τιμές στην λίστα new_patient,
στο εύρος τιμών που κυμαίνεται κάθε χαρακτηριστικό του dataset. Άμα δεν θέλουμε να βάζουμε τυχαίες τιμές και να βάζουμε τα καινούργια δεδομένα
χειροκίνητα ξεσχολιάζουμε τις δύο εντολές κάτω από το εμφολευμένο for loop και σχολιάζουμε την γραμμή val = random.uniform(min(df_dataset[columns[num]]),  max(df_dataset[columns[num]])).
Τέλος καλείται η συνάρτηση k_nearest_neighbors και γίνονται οι προβλέψεις για τους καινούργιους ασθενείς.
"""
def add_new_patient_data_and_predict(df_dataset, dataset, num_of_patients):
    columns = df_dataset.columns
    new_patient = []
    new_patients_list = []
   
    # Άμα θέλω να δίνω τις τιμές χειροκίνητα αρκεί να ξεσχολιάσω τις γραμμές 322 και 323 και να σχολιάσω την 324.
    for i in range(num_of_patients):    
        #print('Add diagnosis == -1\n')
        new_patient.append(-1)
        for num in range(1, 31):
            #print('Add value between ', min(df_dataset[columns[num]]), ' and ', max(df_dataset[columns[num]]), '\n')
            #val = float(input('Enter {} '.format(columns[num])))
            val = random.uniform(min(df_dataset[columns[num]]),  max(df_dataset[columns[num]]))
            new_patient.append(val)    
        
        new_patients_list.append(new_patient)
        new_patient = []

    k_nearest_neighbors(dataset, new_patients_list, 5)







dataset = process_data() # Dataset σε μορφή λίστας που περιέχει λίστες.
study_data()
df_dataset = pd.read_csv('C:/Users/user/Desktop/ERGASIES_&_ARXEIA/Διαχείριση_Γνώσης_2/data_2.csv') # Pandas dataframe.

"""
Τυπώνουμε το min και το max κάθε τιμής.
"""
columns = df_dataset.columns
for num in range(1, 31):
    print('Min value of ', columns[num], ' is', min(df_dataset[columns[num]]), ' and max value is ', max(df_dataset[columns[num]]), '\n')

# Βρίσκουμε τον βέλτιστο αριθμό γειτόνων 
find_best_K(10, dataset)
# Βλέπουμε πως αξιολογείται το μοντέλο μας ανάλογα τον αριθμό των folds.
fold_num_and_accuracy(11, dataset)

df_dataset.dtypes

# Σχεδιασμός διαγραμμάτων ως προς τα χαρακτηριστικά των όγκων και κατάταξη τους σε καλοήθεις και κακοήθεις.
df = pd.read_csv('C:/Users/user/Desktop/ERGASIES_&_ARXEIA/Διαχείριση_Γνώσης_2/data.csv')
sns.lmplot(x = 'radius_mean', y = 'texture_mean', hue = 'diagnosis', data = df)
sns.lmplot(x = 'perimeter_mean', y = 'smoothness_mean', hue = 'diagnosis', data = df)
sns.lmplot(x = 'area_mean', y = 'compactness_mean', hue = 'diagnosis', data = df)

# Κάνουμε προβλέψεις με δεδομένα που ξέρουμε σε ποιο class ανήκουν.
k_nearest_neighbors(dataset, dataset[0:10], 5) # 1 λάθος.
k_nearest_neighbors(dataset, dataset[0:20], 5) # 3 λάθη.
k_nearest_neighbors(dataset, dataset[0:100], 5) #Στα 100 παίρνουμε 10 λάθη.
k_nearest_neighbors(dataset, dataset[0:200], 5) # 16 λάθη.


add_new_patient_data_and_predict(df_dataset, dataset, 10)
add_new_patient_data_and_predict(df_dataset, dataset, 20)
add_new_patient_data_and_predict(df_dataset, dataset, 30) 





 
