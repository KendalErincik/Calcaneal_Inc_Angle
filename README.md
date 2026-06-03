# Automated Calcaneal Inclination Angle (CIA) Measurement

Multi-structure segmentation pipeline for automated CIA measurement from lateral foot X-rays.

## Method

Three anatomical structures are segmented simultaneously using **U-Net + EfficientNet-B0**:

- **Calcaneus inferior surface** → landmarks P2 (lowest point) + P3 (tangent point)
- **5th metatarsal head** → landmark P1 (lowest point)
- **Calcaneocuboid joint** → landmark P4 (inferolateral point)

**CIA = angle(P1→P2, P4→P3)**

### Architecture
- Encoder: EfficientNet-B0 (ImageNet pretrained)
- Decoder: U-Net
- Input: 512x512 grayscale (letterboxed)
- Output: 3-channel segmentation mask
- Loss: Masked Dice + weighted BCE (per-channel validity masking)

### Supported Formats
PNG, JPEG, and DICOM (.dcm)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training
```bash
python calcaneal_cia.py train \
    --images /path/to/images \
    --export /path/to/label_studio_export.json \
    --output model.pth \
    --epochs 60 --batch-size 8 --lr 1e-3
```

### Inference
```bash
python calcaneal_cia.py infer \
    --weights model.pth \
    --images /path/to/test_images \
    --output results/
```

### DICOM to PNG Conversion
```bash
python calcaneal_cia.py convert-dicom \
    --input /path/to/dicoms \
    --output /path/to/pngs
```

## Annotation (Label Studio)

See the Colab notebook for detailed Label Studio setup and annotation instructions.

### Labeling Config (XML)
```xml
<View>
  <Header value="Calcaneal CIA Annotation" />
  <Image name="img" value="$image"
         zoom="true" zoomControl="true"
         brightnessControl="true" contrastControl="true" />
  <PolygonLabels name="poly" toName="img"
                 strokeWidth="3" pointSize="small" opacity="0.7">
    <Label value="calcaneus_inferior"  background="rgba(0,255,200,0.5)" />
    <Label value="metatarsal_5"        background="rgba(255,100,100,0.5)" />
    <Label value="calcaneocuboid"      background="rgba(100,150,255,0.5)" />
  </PolygonLabels>
</View>
```

## License

MIT License
