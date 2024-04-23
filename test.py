import os
import hashlib

def hash_file(filepath):
    """Retourne le hash SHA-256 du contenu d'un fichier."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Lire et mettre à jour le hash en blocs de 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def find_files(directory):
    """Génère un dictionnaire des fichiers et leurs hashes SHA-256 pour un répertoire donné."""
    files_dict = {}
    for root, dirs, files in os.walk(directory):
        for name in files:
            filepath = os.path.join(root, name)
            file_hash = hash_file(filepath)
            files_dict[file_hash] = filepath
    return files_dict

def compare_directories(dir1, dir2):
    """Compare les fichiers de deux répertoires pour trouver des doublons."""
    dir1_files = find_files(dir1)
    dir2_files = find_files(dir2)
    duplicates = {}

    for file_hash, file_path in dir2_files.items():
        if file_hash in dir1_files:
            duplicates[file_path] = dir1_files[file_hash]
    
    return duplicates

# Chemins des deux dossiers à comparer
directory1 = 'backup2/trainSaved'
directory2 = 'backup2/validationSaved'

# Recherche de doublons
duplicates = compare_directories(directory1, directory2)

# Affichage des résultats
if duplicates:
    print("Doublons trouvés entre les dossiers:")
    for dup_path, original_path in duplicates.items():
        print(f"{dup_path} est un doublon de {original_path}")
else:
    print("Aucun doublon trouvé entre les dossiers.")
