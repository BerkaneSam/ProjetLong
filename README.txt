Pour lancer le script de projet long il faut d'abord avoir télécharger MODE-TASK et mettre
le script à la même position que le dossier MODE-TASK
lien MODE-TASK : https://github.com/RUBi-ZA/MODE-TASK

Pour lancer le programme de facon basique :
python3 script_projetlong.py pathxtc pathtop -m raw

pathxtc : la trajectoire xtc à étudié
pathtop : le fichier topologique (pdb) de la protéine à étudié

Options:
-m : choisir la routine a utilisé (mds, pca, t-sne, internal pca ou raw)
-e : nombre d'epoch pour l'apprentissage de l'autoencoder
-ca : option pour ne travailler que sur les carbone alpha pour les raw data
-ag : permet de choisir sur quel groupe d'atomes travaillé(valable pour toutes les routines
sauf raw)
-i : intervalle de découpage du fichier xtc
-pr: choix de la perplexité pour la t-sne