# dark-image-enhancement

## Usage
```
python3 GROUP_RGB_parametric_all.py [GPU_ID] [LEARNING_RATE]
```
- <code>GPU_ID</code> and <code>LEARNING_RATE</code> should be specified.
- Normally, <code>LEARNING_RATE</code> is set at <code>1e-4</code>.

## Function
- <code>DataGenerator</code> helps read images on a batch basis dynamically.
- <code>network</code> defines the network architecture.

## Parameter
- Change <code>RES_DIR</code> to the directory that saves the model and testing images.
