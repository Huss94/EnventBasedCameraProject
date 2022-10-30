%chargement d'un fichier dat, de donnees sous fromat brute.
%file est une chaine de caracteres indiquant nom et chemin du fichier à 
%lire

[Add,Tps]=loadaerdat(file);%Add : adresse brute et Tps=timestamp. 

%conversion en format lisible:
[x,y,pol]=extractRetina128EventsFromAddr(Add);

%x=colonne, y=ligne, pol= polarite.

