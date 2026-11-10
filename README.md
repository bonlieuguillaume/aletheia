# aletheia
Detection of manually tampered and AI-generated SAR images.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bonlieuguillaume/aletheia/blob/main/notebooks/colab_train.ipynb)

## Workflow

```
PC (VS Code)  --git push-->  GitHub  --git pull-->  Colab (GPU)  <-->  Google Drive
  code, configs                                     entraînement        données + checkpoints
```

1. Coder en local, puis `git add . ; git commit -m "..." ; git push`.
2. Ouvrir `notebooks/colab_train.ipynb` dans Colab (badge ci-dessus), GPU activé.
3. Exécuter les cellules : le notebook récupère la dernière version du code, copie le dataset depuis le Drive et lance l'entraînement.
4. Checkpoints (`best.pt`, `last.pt`) et `log.csv` sont écrits sur le Drive dans `MyDrive/aletheia/runs/<config>/`. Si Colab déconnecte, relancer : ça reprend depuis `last.pt`.

## Données

Sur Google Drive, `MyDrive/aletheia/data/sar_dataset.zip` contenant :

```
train/real/  train/fake/
val/real/    val/fake/
test/real/   test/fake/     (optionnel)
```

Formats acceptés : `.tif/.tiff` (y compris complexe SLC → amplitude), `.png`, `.jpg`, `.npy`.
Prétraitement : `log1p` sur l'amplitude puis standardisation par image ; les images sont **découpées** (crop) à `img_size`, jamais redimensionnées, pour préserver les traces de manipulation.

Un seul gros zip plutôt que des milliers de petits fichiers sur le Drive : la lecture fichier par fichier depuis Drive est très lente.

## Modèles

| config | modèle |
|---|---|
| `configs/cnn_baseline.yaml` | petit CNN entraîné from scratch |
| `configs/resnet18.yaml` | ResNet18 pré-entraîné ImageNet, conv1 ramenée à 1 canal |
| `configs/crnn.yaml` | CNN → grille 8×8 de features → BiLSTM (cohérence spatiale entre régions) |

Lancement manuel :

```bash
python -m aletheia.train --config configs/crnn.yaml --data-root /content/data --output-dir /content/drive/MyDrive/aletheia/runs
```

## Structure

```
src/aletheia/data.py          chargement / normalisation des images SAR
src/aletheia/models/          CNN, ResNet, CRNN
src/aletheia/train.py         boucle d'entraînement (AMP, reprise, AUC)
configs/                      hyperparamètres
notebooks/colab_train.ipynb   lanceur Colab
```
