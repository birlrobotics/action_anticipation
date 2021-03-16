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
- [ ] 对子动作segment的长度回归可能**采用范围值而非确定值（或学习一个offset）**会好点，因为子动作持续时间是浮动的；
- [ ] **多尺度未来特征生成**，因为小物体在浅层特征才具有一定分辨率；
- [ ] 在transformer中的attention使用**多尺度attention/local attention**而不是全局attention；
- [ ] 一开始从I3D得到的特征是否需要通过MLP后作动作识别；
- [ ] 逆视频输入，加个可学习的正逆特征，像PE特征一样；
- [ ] 初始特征增强，加噪声

### NOTE
- For simplicity, we sample the training data in each pure action segment, while we construct the evaluation datas with continuous frames.


### Log
- lr = 0.0001时，训练会明显震荡，检查数据好像没啥问题，**改成0.00001会好点**；
- 所有层使用**default初始化**比xavier_uniform好；
- input feature先试用L2norm归一化或L2norm，好像也没啥作用；
- **batchsize设小点**好像效果更好；
- 没有Positional embedding效果好差;
- I3D features没有经过FC效果会差点,**加两层MLP会好点**；
- dropout in PE效果不好；
- decoder中PE加offset而不是从0位置计起，效果没提升；
- 4层效果比两层好;

CUDA_VISIBLE_DEVICES=1 python train.py --nw=4 --lr=0.00001 --bs=4 --e_v='L2_dp0.3_lr0.00001_bs64_dfinit_alldata_inputl2norm'