# Doserad challange

## Todos:
 - Figure out what is the best spacing configuration
 - Find out hierarchy of training ( beam level, ray level, beamlet level)


## Training structure
### Metadata encoding
#### Encoding extra metadata proton/SynthRad...
- One way to encode metadata or multiple models.
- What metadata to encode?
- Does it even help?

#### Encoding base metadata
- Should we use MLP encoding?
- Geometric field encoding. (Source and target encoding)
- Fourirer encoding.
- Postional encoding (sine and cosine)

### Preprocess
#### Training level
- Should we train at beam level, ray level or beamlet level?
#### Data augmentation
- Should we train on HU or RSP?
- What is the ideal size of input volume? (spacing, target size)

### Losses.
- Use only competition default loss?
- Use combinations?
- If latent transition, should we use extra regularizations?

## Model architectures.
- Vision Transformer.
- Conv encoder + Transformer transition + Conv decoder.
- 3D Unet.
