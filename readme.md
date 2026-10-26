# Parallel Prediction Model Based on Informer + TCN-SENet

This issue proposes an Informer+TCN-SENet parallel prediction model based on a wind power power data set, which significantly improves the prediction effect in single step prediction tasks!
● Dataset: Wind power dataset, power data set, wind speed dataset, etc. of a wind farm
● Environment Framework: Python 3.9, PyTorch 1.8 and above can be run.
● Step-by-step prediction model score: 0.98 on test set
● Usage Object: Paper requirements, graduation design requirements.
● Code Guarantee: The code comments are detailed and ready to run.
We also provide experiments and comparisons based on multiple datasets in multiple prediction tasks, and provide detailed documentation and tutorials including how to replace your own dataset, parameter tuning tutorials, and replacement of prediction tasks. The code is commented line by line, with detailed introductions to parameters.
\tEffectiveness comparison:
TCN prediction model:
Model Evaluation:
\tPredictive Visualization:
\t  
\tInformer + TCN-SENet parallel prediction model:
Model Evaluation:
\tPredictive Visualization:
1. Introduction to Model Innovation Points
1.1 Combine the advantages of Informer and TCN-SENet
Informer: Expert in handling long sequences, capable of parallel computation, enhancing computational efficiency and predictive performance. Informer has been improved on the basis of Transformer to better suit time series data, especially those with long-term dependencies.
TCN-SENet: Utilizing a series of time convolutional layers from the Time Convolutional Network (TCN), these convolutional layers can effectively capture sequence patterns at different temporal scales. Applying the SENet module on the output of TCN to learn and enhance the representation of important features, enhancing the model's ability to perceive key information.
By parallelly using these two models, it is possible to better capture patterns at different temporal scales, thereby enhancing the accuracy and robustness of predictions.
1.2 Parallel Model Architecture
By concurrently using Informer and TCN-SENet, learning on two separate branches allows the model to extract and process information at various temporal scales:
(1) The Informer part:
Primarily designed for capturing global temporal patterns, suitable for handling longitudinal sequence data.
Enhanced the ability to capture distant temporal dependencies through a sparse self-attention mechanism.
Global feature extraction for the input data.

(2) TCN-SENet:
Primarily used for capturing local spatial patterns, capable of handling short-term dependencies and dynamic changes in sequence data.
Utilizes convolutional structures provided by TCN to capture the short-term dynamics of sequences.
The SENet module enhances the representation capability of feature channels, adaptively adjusting the importance of different channels.

This architecture comprehensively captures the characteristics of time series data, improving the model's predictive performance.
1.3 Model Fusion
By concatenating the outputs of Informer and TCN-SENet with a fully connected layer to fuse features from different models, this approach enables the model to simultaneously utilize the global information extraction capabilities of Informer and the local spatial relationship modeling capabilities of TCN-SENet.
1.4 Efficient Computation
The use of Informer significantly improves the computational efficiency for longitudinal sequence processing, while TCN leverages the causality and parallelism of convolution operations, often faster than RNNs, especially in long sequences. This combination ensures efficient computation while enhancing prediction accuracy and reliability.
2 Informer Demonstration, Three Key Innovations
The probability sparse self-attention is a sparse self-attention mechanism introduced in the Informer model. Its core idea is to select the most important part of attention weights through probability methods, while ignoring those weights that have a smaller impact on the results. This method can significantly reduce computational complexity while maintaining high model performance.
Sparse self-attention: Unlike the dense self-attention mechanism of standard Transformers, Informer introduces ProbSparse Self-attention, which selects important Q-K pairs through a probability sampling mechanism, reducing computational complexity.
Efficiency improvement: The sparse attention mechanism significantly reduces computational complexity from O(L2⋅d) to O(L⋅log(L)⋅d), where L is the sequence length and d is the feature dimension per time step.
Multi-scale feature extraction - Information distillation
The architecture diagram of Informer does not mark N Encoder stacks on the left side as in Transformers, but consists of two large and two small trapezoids.

Upon horizontally viewing an individual Encoder (i.e., the large trapezoid on the left side of the architecture diagram, representing the main stack of the entire input sequence).

The Encoder's role is Self-attention Distilling. Since many ProbSparse self-attention mechanisms use V's mean for filling, there naturally exists redundant attention sources. Therefore, convolution and pooling are applied between adjacent Attention Blocks to downsample features. Thus, when designing the Encoder, distillation operations are continuously extracted to focus on key features, thereby obtaining feature maps worth paying attention to.
Multi-scale Time Series Feature Extraction: Informer models features at different temporal granularities through multi-scale approaches, effectively capturing the multi-scale dependencies in time series.
Information distillation: Utilizes an information distillation mechanism, progressively reducing the temporal step size through hierarchical time-convolutional pooling layers to extract features at different scales, achieving efficient modeling over long temporal dependencies.
Convolutional Dimensionality Reduction: Using a 1D convolutional pooling layer in the encoder for dimensionality reduction, with a stride of 2 to half the sequence length, further reducing computational complexity.
Information compression: Through convolutional pooling layers, information is compressed, condensing long sequences of data into shorter time steps, thereby enabling more efficient temporal modeling.
2.3 Time coding
\tInformer\tNot only has the Transformer architecture's essential Position Embedding been added to the original vector, but various encodings related to time have also been added:
\tDay cycle code: indicates a time point in a day.
\tWeekly cycle code: Represents a point in time within a week.
Month cycle code: Represents a point in time within a month.
In LSTF issues, the ability to capture remote independence requires global information such as hierarchical timestamps (week, month, year) and unknowable timestamps (holidays, events).
The specific addition of what kind of GlobalTimeStamp is needed to be added here still needs to be confirmed according to the actual problem. If calculating the passenger flow at high-speed train stations, it is evident that the "holiday" time difference is very important. If calculating the passenger flow of public transportation such as buses and subways, it is clear that "weekday" can reveal whether it is a working day.
3. Data Introductions on Wind Power Power and Other Datasets
3.1 Import Data
\t  
The wind power dataset consists of 35,040 samples with 15 features. The first 6,000 samples are selected for visualization.
3.2 Dataset Preparation and Preprocessing
For detailed information, please refer to the provided documents!
\t  
4. Parallel Prediction Model Based on Informer-TCN-SENet
\t4.1 Define the parallel prediction network model based on Informer-TCN-SENet
\t4.2 Set parameters and train the model
\t50 epochs, MSE is 0.01283, the performance of the Informer-TCN-SENet parallel prediction is significant, the model can fully utilize the long-term dependency modeling ability of Informer and the spatial dependency capturing ability of TCN-SENet, converge quickly, superior performance, high prediction accuracy, appropriately adjusting model parameters can further improve the model's predictive performance.
5. Results Visualization and Model Evaluation
\t5.1 Prediction results visualization
\t5.2 Model evaluation
\tFrom the prediction results, it can be seen that the fitting effect is good under the Informer-TCN-SENet parallel prediction model, through this design, we can fully utilize the advantages of Informer and TCN-SENet to achieve efficient and accurate time series prediction, the combined prediction effect is significant!
6. Tuning Parameters for Informer-TCN-SENet Parallel Model
\tIn developing and tuning the parallel prediction model based on Informer-TCN-SENet, tuning parameters is a key step. Through reasonable parameter adjustment, the performance of the model can be significantly improved. See the detailed tutorial in the provided document!
\t6.1 Model architecture parameters
\tInformer part:
\td_model: The hidden layer dimension of the model. Generally starts from 128 or 256, gradually adjusting to 512 or even higher.
\tn_heads: The number of attention heads. Common values are 4 or 8, which can be adjusted according to the hidden layer dimension.
\tn_layers: The number of encoding layers and decoders in the Informer. Generally set between 2-6 layers, more layers may enhance the model's expressive power but also increase computational overhead.
TCN-SENet Part\tnum_channels: The number of output channels in each TemporalBlock, generally starting from 32 or 64, and can be adjusted as needed.
\tkernel_size: Kernel size, generally set to 3.
6.2 Training Parameters\tbatch_size: The size of the batch for each training iteration. Common values are 32, 64, or 128, adjusted according to memory size and model complexity.
\tlearning_rate: The learning rate. Initially, it is generally set to 1e-3 or 1e-4, which can be dynamically adjusted using a learning rate scheduler.
\tepochs: The number of training rounds. Typically starts from 50-100, adjusted according to the model's convergence.
6.2 Regularization and Optimizers\tdropout: Dropout rate. Common values are 0.1-0.3, used to prevent overfitting.
\tweight_decay: Weight decay coefficient. Common values are 1e-4, used for L2 regularization.
\toptimizer: Optimizer. Commonly used are Adam or AdamW, but other optimizers such as SGD can be tried.
Click to download: Complete original data, Python code
Previously Featured Content:
Time Series Forecasting: Analysis and comparison of LSTM, ARIMA, Holt-Winters, SARIMA models
Wind Speed Prediction (Part 1) Data Set Introduction and Preprocessing
Wind Speed Prediction (II) EMD-LSTM Model Based on Pytorch
Wind Speed Prediction (III) EMD-LSTM-Attention Model
Wind Speed Prediction (IV) EMD-Transformer Model Based on Pytorch
Wind Speed Prediction (5) EMD-CNN-LSTM Model Based on Pytorch
Wind Speed Prediction (VI) - EMD-CNN-GRU Parallel Model Based on Pytorch
CEEMDAN + Combination Prediction Model (BiLSTM-Attention + ARIMA)
CEEMDAN + Combination Prediction Model (CNN-LSTM + ARIMA)
CEEMDAN + Combination Prediction Model (Transformer - BiLSTM + ARIMA)
CEEMDAN + Combination Prediction Model (CNN-Transformer + ARIMA)
Multi-Feature Variable Sequence Forecasting (I) - CNN-LSTM Wind Speed Forecasting Model
Multi-Feature Variable Sequence Forecasting (II) - CNN-LSTM-Attention Wind Speed Forecasting Model
Multi-Feature Variable Sequence Forecasting (III) - CNN-Transformer Wind Speed Forecasting Model
Multi-Feature Variable Sequence Forecasting (IV) Transformer-BiLSTM Wind Speed Forecasting Model
Multi-Feature Variable Sequence Forecasting (V) CEEMDAN+CNN-LSTM Wind Speed Forecasting Model
Multi-Feature Variable Sequence Forecasting (VI) CEEMDAN+CNN-Transformer Wind Speed Forecasting Model
Prediction Model Based on the Particle Swarm Optimization Algorithm SSA - CEEMDAN-BiLSTM-Attention
Prediction Model Based on the Particle Swarm Optimization Algorithm SSA - CEEMDAN-Transformer-BiGRU
Superior Prediction Model: Secondary Decomposition-Combination Prediction
Multi-Feature Variable Sequence Forecasting (VII) CEEMDAN+Transformer-BiLSTM Forecasting Model
Multi-Feature Variable Sequence Forecasting (VIII) Prediction Model Based on the Particle Swarm Optimization Algorithm for CEEMDAN-SSA-BiLSTM
Multi-Feature Variable Sequence Forecasting (11) Prediction Model Based on Pytorch's TCN-GRU
VMD + CEEMDAN Secondary Decomposition, CNN-LSTM Forecasting Model
VMD + CEEMDAN Secondary Decomposition, CNN-Transformer Prediction Model
VMD + CEEMDAN Secondary Decomposition, Transformer-BiGRU Prediction Model
Prediction Model Based on the Particle Swarm Optimization Algorithm SSA - Complete Code Kit
Superior Combination Prediction Model!
High Innovation | CEEMDAN + SSA-TCN-BiLSTM-Attention Prediction Model
Superior Combination Prediction Model!
Lowest Price Online | Continuous Updates for the Complete Kit!
Based on TCN-SENet + BiGRU-GlobalAttention Parallel Prediction Model
Secondary Decomposition - Innovative Model Prediction Complete Kit
BiTCN-BiGRU-CrossAttention Fusion of Spatiotemporal Features High Innovation Prediction Model
Updated Combination Prediction | Superior Combination Prediction Model!
Spatiotemporal Feature Fusion BiTCN-Transformer Parallel Prediction Model
Based on Multi-Level Attention Mechanism Parallel Prediction Model
CEEMDAN-CNN-GRU-GlobalAttention + XGBoost Combination Prediction
Multi-Step Prediction Series | LSTM, CNN, Transformer, TCN, Serial, Parallel Model Collection
End
\t2. Friends who like can click to follow, give "Like" to the article, share on WeChat groups or discuss in groups

## Images

<!-- PAYMENT_BLOCK -->

## Get this dataset

- Payment link (Stripe): [Buy now](https://buy.stripe.com/3cs8yP7sY87d0vu9AB)
- Payment QR code: ![Payment QR code](pay.png)

<!-- /PAYMENT_BLOCK -->
