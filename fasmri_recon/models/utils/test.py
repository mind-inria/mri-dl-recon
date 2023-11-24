import tensorflow as tf

# Création de deux tenseurs, k_data et mask
k_data = tf.constant([[1.0, 2.0, 3.0],
                     [4.0, 5.0, 6.0]])

mask = tf.constant([[0, 1, 0],
                   [1, 1, 0]])

# Combinaison des deux tenseurs en une paire
input_pair = (k_data, mask)

# Utilisation de la fonction _mask_tf
result = _mask_tf(input_pair)

# Affichage des résultats
print("k_data :\n", k_data.numpy())
print("mask :\n", mask.numpy())
print("Résultat de la multiplication élément par élément :\n", result.numpy())

