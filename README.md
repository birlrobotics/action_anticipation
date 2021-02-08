### Data processing
`` python -m dataset.video2frame './dataset/breakfast/original/' './dataset/breakfast/rgb_frame/' --nw=12 ``

`` python -m dataset.annotation_gather './dataset/breakfast/original/' './dataset/breakfast/' ``

### TODO
- [x] When processing the stereo video, it seems we just need to choose the ch0 video;
- [ ] Add data augmentation;
- [x] Sample the clips based on the GT labels to guarantee the integrity of the video;
- [x] Implement the positional encoding in Transformer;
- [ ] Which normalization we should use in Transformer;
- [ ] Shall we need the ReLU activation function when calculating the (Q, K, V) in attention;
- [ ] shall we need a Conv layer in I3D head instead of average pooling?

### IDEAS
- [x] Transformer