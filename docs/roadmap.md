# ⏰ RoadMap
## ✅ Phase 1: Complete OpenSeek-data-1.3TB Creation & OpenSeek-Small Distributed Training 
### 📊 Data
- [x] Build data processing and synthesis pipeline
- [x] Build OpenSeek-PTx1.3T-v0.1
- [x] Construct OpenSeek-data-1.3T official version based on OpenSeek-Small data ratio experiments

### 🔄 Training
- [x] Validate 3B model effects on OpenSeek-PT-1.3T-v0.1 (Baseline)
- [x] Complete experimental training of OpenSeek-Small (~100B)

### 💻 System
- [x] Support distributed training for MLA, DeepSeek MoE, MTP, Auxiliary-Loss-Free etc.
- [x] Convert and load DeepSeek V3 parameters

## ⚡ Phase 2: Expand Data Scale & Optimize Distributed Training Performance
### 📊 Data
- [x] Expand data scale, build OpenSeek-PT-8T
- [x] Construct Long-CoT-Backward synthetic dataset and verify effects

### 🔄 Training
- [ ] ⚡ Complete hyperparameter experiments for OpenSeek-Small
- [ ] ⚡ Validate OpenSeek-PT-8T effects
- [ ] ⚡ Complete full training of OpenSeek-Small on OpenSeek-PT-1.3T-v1.0

### 💻 System
- [ ] ⚡ Support Node-limited Routing MoE
- [ ] ⚡ Support FP8 distributed training
- [ ] ⚡ Integrate Triton-based operator library FlagGems

## Phase 3: Support Larger Scale Data & Distributed Training
### 📊 Data
- [ ] Build OpenSeek-Zero dataset
- [ ] Build OpenSeek-RL dataset
- [ ] Build OpenSeek-SFT dataset
- [ ] Construct Long-CoT-Forward synthetic dataset and verify effects

### 🔄 Training
- [ ] Produce OpenSeek-Small-Zero
- [ ] Produce OpenSeek-Small-SFT
- [ ] Produce OpenSeek-Small-RL
- [ ] Complete hyperparameter experiments for OpenSeek-Mid
- [ ] Complete full training of OpenSeek-Mid on OpenSeek-PT-8T

### 💻 System
- [ ] Support DualPipe pipeline parallelism
- [ ] Further optimize computation-communication overlap and memory optimization

## Phase 4: Upgrade Multi-chip Support & Open Source Release
### 📊 Data
- [ ] Release official version of OpenSeek series datasets
- [ ] Construct Long-CoT-RAG synthetic dataset and verify effects

### 🔄 Training
- [ ] Produce OpenSeek-Mid-Zero
- [ ] Produce OpenSeek-Mid-SFT
- [ ] Produce OpenSeek-Mid-RL

### 💻 System
- [ ] Adapt training and precision alignment for different chips
- [ ] Implement customized parallel and optimization strategies for specific chips

