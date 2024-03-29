# Literatures
- Prognosis of a Wind Turbine Gearbox Bearing Using Supervised Machine Learning
- Prediction of Wind Turbine Generator Bearing Failure Through Analysis of High Frequency Vibration Data and the Application of Support Vector Machine Algorithms
- Analyzing Bearing Faults in Wind Turbines: A data-mining Approach
- An Insight into Wind turbine Planet Bearing Fault Prediction Using SCADA Data
- Failure and Remaining Useful Life Prediction of Wind Turbine Gearboxes
- Automatic Fault prediction of Wind Turbine Main Bearing Based on SCADA Data and Artificial Neural Network
- Diagnostic Models for Wind Turbine Components Using SCADA Time Series Data
- Exploiting SCADA System Data for Wind Turbine Performance Monitoring
- Wind Turbine Gearbox Failure and Remaining Useful Life Prediction using Machine Learning techniques
- Machine Learning for Long Cycle Maintenance Prediction of Wind Turbine
- LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection
- Diagnosing Wind Turbine Faults Using Machine Learning Techniques Applied to Operational Data
- Use of SCADA Data for Failure Detection in Wind Turbines
- Learning Deep Representation of Imbalanced SCADA Data for Fault Detection of Wind Turbines
- Monitoring Wind Turbines Using SCADA
- Wind Turbine Fault Diagnosis and Predictive Maintenance Through Statistical Process Control and Machine Learning
<br/>

# Summary

## [Prognosis of a Wind Turbine Gearbox Bearing Using Supervised Machine Learning](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6679281/)

- Objective: Predict the RUL of HSS Bearing
- Data: Vibration Measurements
- Root Mean Square(RMS), Kurtosis(KU), and Energy Index(EI) were analyzed to define the bearing failure stages
- combines two supervised machine learning techniques: Regression model and multilayer ANN
- Conducted a comparative study of two regression models: Polynomial and Exponential, Chose the one with best fitting performance to fit the condition indicators(eg. KU, EI, RMS) extracted from the data
- Identification of a suitable indicator simplifies the degradation assessment and prognostics. 
  - Monotonicity and trendibility are a type of metrics used to quantify the indicators suitableness.
  - Monotonicity: Indicates whether or not the sequence is increasing or decreasing. Since bearing degradation is irreversible process, monotonicity measures if condition indicator is suitable for degradation or not.
  - Trendibility: Indicates the degree to which the condition indicator values at diff time have the same fundamental shape. 
- Regression model's results were fed to the feed-forward back-propogation Neural Network to enable better predictions. 
- NN: One input layer with two inputs(Raw RMS, KU), One output layer with one output(fitted RMS), two hidden layers with 9 and 7 neurons respectively
- ANN performs the best

## [Prediction of Wind Turbine Generator Bearing Failure Through Analysis of High Frequency Vibration Data and the Application of Support Vector Machine Algorithms](https://strathprints.strath.ac.uk/66304/)

- Objective: Classify the turbine bearing between healthy and failure (1-2 months before failure).  
- Data: Vibration Data
- Results show that by analysing high frequency vibration data and extracting key features to train support vector machine algorithms, an accuracy of 67% can be achieved in successfully predicting failure 1-2 months before occurrence. 
- If too many different examples are considered of different wind turbines and operating conditions the overall accuracy can be diminished.
- Vibration Analysis Techniques:
  - Time Domain Features: Minimum, Maximum, Mean, RMS, Standard Deviation, Kurtosis
  - Frequency Domain Features: 1P Amplitude, 2P Amplitude, 1P Energy, 2P Energy
  - Order Domain Features: 1st Order Amplitude, 2nd Order Amplitude

## [Analyzing Bearing Faults in Wind Turbines: A data-mining Approach](https://www.sciencedirect.com/science/article/pii/S0960148112002613)

- Objective: Prediction of over temperature events in bearing using Neural Networks. Resulted in 97% accuracy and prediction of faults 1.5 hour before their occurance.
- Data: High frequency (10 s) SCADA data
- Selected parameters using Domain knowledge and Data mining algorithms
- Metrics: Absolute Error, Mean Absolute Error, Relative Error, Mean Relative Error, Coefficient of Determination (R-squared)
- Selected best 4 NNs. The NNs were generated by varying the number of neurons in hidden layer and activation functions. 
  - The number of neurons was kept in the range of 5-25 
  - Activation functions: tanh, exponential, identity and logistic
  - Network structures were optimized using BFGS method
- The error residuals were analyzed using an average window of size 360.

## [An Insight into Wind turbine Planet Bearing Fault Prediction Using SCADA Data](https://pdfs.semanticscholar.org/c5a8/2aa6c09c0d6e85c6ee78accff00a5198d7a0.pdf) 

- Objective: Predict the failure of a wind turbine gearbox planet bearing before 12 months, 6 months, 1 month. 
- Data: 10 minute averaged SCADA data from 3 operating wind turbines that have a double planetary stage gearbox
  - Historic data is collected for more than a year at sparse time periods before the occurance of a bearing failure ona planet of the first planetary stage.
- A cluster filter is applied on the training data and aims to remove outliers depending on the operating conditions of the wind turbine.
  - The distance is calculated for each data vector in the trainng data set from its cluster center. The Mahalanobis distance values can be estimated by a loglogistic distribution function and data below the probability threshold 2.5% are filtered out.
- As the gearbox fault gets worse, the rotor frequency/slip/speed algorithm is incorrectly picking up a faulty harmonic, rather than the slip frequency.
- Robust linear regression is performed between generator speed and rotor speed. Results shows that within 1 month before failure, the speed is underpredicted.
- Fault Detection: The performance of the normal behaviour model is assessed using the distribution of errors, which in a healthy operation should have a mean around zero. If an abnormality occurs, the behaviour prediction model should yield higher errors and therefore the mean will be shifted.
  - Performed t-test which resulted in rejection at significance level 1% during the last month of the bearing operation since the regression error means of the training and testing sets differ by a large amount.
- Performed multiclass and binary classification using Support Vector Machine with radial kernel which resulted into 94.7% and 98.7% accuracy.

## [Failure and Remaining Useful Life Prediction of Wind Turbine Gearboxes](https://pdfs.semanticscholar.org/ae08/2a0eaa2f6ee24b0c51b898291840b17151c5.pdf)

- Objective: Predict Wind Turbine Gearbox incipient faults using a combination of condition monitoring data
- Data: SCADA data + Vibration Data
- The variable loading conditions of wind turbines and the nonlinear relationships between the parameters render rule-based monitoring impractical. 
- The temperature in the hollow shaft is predicted using neural networks, based on other temperatures, the power and speed of turbine. The mean absolute error of predicted temperature increases towards failure. System health assessment can be easily achieved but gearbox specific component fault detection can be more challenging.
- The vibration analysis methods involve transforming the signals into the frequency/order domain or time frequency domain where fault signatures are revealed more clearly. The features extracted based on these fault signatures, are used as inputs in pattern recognition models that can determine the health state of the components. Results are evaluated based on the robustness, missed detection rate and how earlier faults can be detected compared to rule-based methods which are the current practice.
- Feature extracted from vibrations can be ranked and fused to create health index. RUL is estimated using degradation models. - Fleet based diagnostics and prognostics can provide a robust fault detection framework based on similar wind turbines. Assuming the majority of wind turbines in the fleet are in normal operating condition, a clustering approach can identify baseline data and detect abnormality.

## [Automatic Fault prediction of Wind Turbine Main Bearing Based on SCADA Data and Artificial Neural Network](https://www.researchgate.net/publication/326013496_Automatic_Fault_Prediction_of_Wind_Turbine_Main_Bearing_Based_on_SCADA_Data_and_Artificial_Neural_Network)

- Objective: Proposes a methodology of fault prediction and automatically generating warning and alarm for wind turbine main bearing based on stored SCADA data using Artificial Neural Network. The ANN model of turbine main bearing normal behavior is established and then the deviation between estimated and actual values of parameter is calculated.
- Data: SCADA data
- To avoid false warning/alarm, a time interval (i.e. a week) is used to calculate the percentage of time that the deviation is above warning/alarm threshold. 
- This paper looks the fault detection as a regression problem to estimate relationships between key condition indicator and other performance parameters.
- Output: Rear Bearing Temperature (t)
- Input: Rear Bearing Temperature (t-1), Active power output (t, t-1), Ambient Temperature (t, t-1), Turbine Speed (t, t-1)
- In the training process: The measurement of turbine rear bearing temperature (t-1) is not used as an input but uses previous estimated temperature (t-1) instead to avoid abnormal temperature afecting to ANN model. 
- Instead of setting warning/alarm at each data point, a time interval with sliding window is used to judge if the warning or alarm will be generated. If there are 25% production time with deviation over fixed warning/alarm level, the system will generate warning/alarm to operator automatically.

## [Diagnostic Models for Wind Turbine Components Using SCADA Time Series Data](https://www.nrel.gov/docs/fy18osti/71166.pdf)

- Objective: An unsupervised approach to detect significant failures based on predetermined criteria- Abnormal spikes in turbine component temperature followed by turbine shut-off
- Data: SCADA Data
- Developed a model that adjusts component temperature for weather and power production returning a normalized temperature value.
- Finding out the residuals from observed component temperature and estimated component temperature by independent variables (Ambient Temperature, Power Output) and flagging them as a failure
- Criteria: Root Mean Square Error (RMSE), The pearson correlation coefficient (PCC), and Shapiro-Wilk Normality test (SWNT)
  - RMSE: measures the fit of the model
  - PCC: To track how much of the effect ambient temperature and power has been eliminated from the raw data
  - SWNT:to see how much a certain sample differs from the normal distribution
- Tested models: Linear Regression, Multivariate Polynomial Regression, Random Forest, Neural Networks
- Lowest RMSE: Multivariate Polynomial Regression, Lowest PCC: Linear Regression

## [Exploiting SCADA System Data for Wind Turbine Performance Monitoring](https://www.researchgate.net/publication/261041896_Exploiting_SCADA_system_data_for_wind_turbine_performance_monitoring)

- Objective: Model the turbine power output of each turbine during fault-free operationand to subsequently use the trained model to identify performance degradation by analyzing the residual between the predicted and observed power values for each turbine.
- Data: SCADA Data
- Wind turbine power curve analysis is a common method for providing a universal measure of wind turbine performance and as an indicator of overall wind turbine health.
- To model the relationship between model inputs and turbine power production, Gaussian process (GP) models were used. 
  - A motivating factor for using GP models is that this technique describes each model prediction in terms of Gaussian Distribution, described by mean and a variance value.
  - The width of the confidence limit generated by a GP reflects how well the training data describes the relationship between the model input and the model output.
- Input: Wind Speed and Air Density (10 minute average)
- To identify performance degradation in turbines, the cumulative sum of power residual, generated at each time step, was computed over time for test period.

## [Wind Turbine Gearbox Failure and Remaining Useful Life Prediction using Machine Learning techniques](https://onlinelibrary.wiley.com/doi/full/10.1002/we.2290)

- Objective: Prediction of failure and remaining useful life using machine learning techniques (Neural Networks, SVM, Logistic Regression)
- Data: SCADA and Vibration Data, O&M Orders, Logs
  - using SCADA data failure can be predicted up to a month before it occurs
  - using Vibration data failure can be predicted 5 to 6 months in advance
- Two classes neural network can correctly predict gearbox failures between 72.5% to 75% of time depending on the failure mode when trained with SCADA data and 100% of time when trained with vibration data
- "Permutation Feature Importance" (PFI) function was used to get an importance for each input in the prediction. The evaluation metric was chosen to be the accuracy of the learning algorithm. The PFI function computes the sensitivity of a model in terms of evaluation metric to random changes of the input values.
- Test data: data from less than 1 month before failure were used as failure and greater than 1 year before failure were used as Healthy

## [Machine Learning for Long Cycle Maintenance Prediction of Wind Turbine](https://www.ncbi.nlm.nih.gov/pubmed/30965619)

- Objective: Presents a method based on machine learning to predict long cycle maintenance time of wind turbines for efficient management in the power company.
- Data: Sensor data including operation data, maintenance time data and event codes collected from 31 wind turbines in two wind farms

- Methodology:
  - Data aggregation is performed to filter out some errors and get significant information from data
  - All maintenance periods are firstly divided into four categories and relations of the maintenance periods of these categories are analyzed.
  - Apriori algorithm is used to find the key event codes, which occur before the maintenance period
  - Linear regression (Wind speed and output power) is applied to filter some errors from the operation data (filter if residual is greater than the threshold) and used to model the relationship between operation data
  - Six features: Wind speed, the power output of wind turbines, the oil temperature of wind turbine gearbox, the temperature of the high speed bearings, the number of consecutive short under-maintenance periods, frequency of the event codes 
  - Hybrid network is built to train the predictive model based on the convolution neural network and support vector machine where the output of the CNN is replaced by SVM
    - CNN is quite good at learning invariant features but not always optimal for classification
    - SVM with a fixed kernel cannot learn complicated invariances but can produce good decision surfaces when applied to well behaved feature vectors
- Prediction accuracy is higher than 70% for the long cycle maintenance within 100 days 
  
## [LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection](https://arxiv.org/abs/1607.00148)

- Objective: Detect Anomalies using reconstruction errors from Long Short Term Memory Netowrk based Encoder-Decoder (EncDec-AD)
- Data: Publicaly available time-series dataset- Power Demand, Space Shuttle and ECG, and two real world engine datasets
- Presents EncDec-AD is robust and can detect anomalies from predictable, unpredictable, periodic, aperiodic and quasi-periodic time series. It is also able to detect anomalies from short as well as long time-series
- intuition: model is trained on normal instances and learn to reconstruct them. When given an anomalous sequence, it may not be able to reconstruct it well and will lead to higher construction error
- Error vectors are used to estimate mean, sd of a normal distribution using maximum likelihood estimation and then anomaly score of any point can be calculated.

## [Diagnosing Wind Turbine Faults Using Machine Learning Techniques Applied to Operational Data](https://ieeexplore.ieee.org/document/7542860)

- Objective: Recognize fault and fault-free operation 
- Data: SCADA data of a turbine in the South-East of Ireland
  - Fault and Alarm data is filtered and analysed in conjuction with the power curve to identify periods of nominal and fault operation
- Three level of classification
  - Fault/No-Fault Operation
  - Classifying a specific fault
  - Advance prediction of specific fault
- Dveloped SVM model using Scikit-Learn's LibSVM implementation. 
  - Subset of 30 features out of 60+ features were selected
  - A randomized grid search was performed over a number of hyperparameters used to train each SVM to find one which yield the best results
  - Verfied using 10-fold cross validation (Scoring metric: mean of weighted precision and recall)
  - three kernels: linear, radial-basis (Gaussian), polynomial
  - For imbalance: assigning class weightage and undersampling

## [Use of SCADA Data for Failure Detection in Wind Turbines](https://www.nrel.gov/docs/fy12osti/51653.pdf)

- Objective: Develop Anomaly Detection algorithm and investigate classification techniques using clustering algorithms and principal componenet analysis for capturing fault signatures
- Data: SCADA
- Used Non-linear PCA by auto-associative neural network approach to extract useful and non-redundant information from sensor data
  - AAAN processes the input data through five sequential layers, one input layer, three hidden layers composed of a mapping layer, a bottleneck layer, and a de-mapping layer, and one output layer
  - this network learns an approximation of the identity mapping between inputs and outputs
  - Q and T-squared statistics are computed. 
    - The Q statistic is a measure of the amount of variation not captured by model
    - T-squared statistic is a measure of the variation in model
- Self Organizing feature maps were used to find patterns in the data
  - It is an unsupervised clustering algorithm that forms neurons located on a regular grid, usually in one or two dimensions
- Both of the automated fault diagnostic algorithm were successful in producing persistent indicators that are well distinguished from those generated based on no-fault data

## [Learning Deep Representation of Imbalanced SCADA Data for Fault Detection of Wind Turbines](https://www.sciencedirect.com/science/article/pii/S0263224119302386)

- Objective: To preserve within-class information and between class information based on triplet loss by learning deep representation using deep NN and predict blade icing accretion fault
- Data: SCADA data
- Target is to learn a Euclidean embedding g(x) from a data sample x into a another feature space based on DNN
- Triplet Score:
  - has three components
    - an anchor
    - one data sample having same class with anchor
    - one data sample having opposite class with anchor
  - This loss function minimizes distance between anchor and positive, while the distance between anchor and negative is enlarged
- A bypass component is designed exquisitely and introduced to the whole network which contains the information flow of gloabal features of each SCADA data sample
- metrics: preicision, recall, f1 score

## [Monitoring Wind Turbines Using SCADA](https://www.windtech-international.com/editorial-features/monitoring-wind-turbines-using-scada)

- Objective: Identify a change in the performance of a Wind Turbine Generator
- Data: SCADA 
- Presents 4 methods to evaluate performance of WTGs over time using power, wind speed and ambient temperature SCADA measurements
- In each method, KPI is calculated which are useful to identify changes or trends in the turbine operation, for performance improvement and in detection and prevention of possible failures in components.
- An algorithm to automatically identify changes in the KPIs is also presented.
- Selecting and filtering of the SCADA Data
  - Only periods with no alarm reported and 'time ok' counter greater than 595 seconds are considered
  - Manual filtering is done by plotting power curve and eliminating the outlier points
- (1) Power Residual Method:
  - Based on the comparison between the power measurements of the WTG and the power obtained from a model of the operation of the WTG
  - The power residuals are then monitored over time, looking for deviations in the difference
  - Mitigate variability of the residuals, simple moving average and exponentially weighted moving average are calcualted and monitored.
- (2) Health Value -PC2 Dev Method:
  - Second Eigen value of power and wind speed data is evaluated in time windows of one week
  - The first eigenvalue represents the variability along the main direction of the scatter of power-velocity, while the second eigen value shows variations in the direction transversal to scatter
  - As ambient temperature is not an input to this model, this KPI may present variations due to seasonality
- (3) Quantiles Method:
  - Aim of this KPI is to evaluate over time the power production of the WTG regardless of the measured wind speed value at each 10 minute period.
  - Filtering power data for each wind speed occuring and binned and sorted from higher to lower assigning a quatile value
  - Due to high variablity of quantiles moving average is calculated and monitored
- (4) Power Curves Evolution Method:
  - This algorithm consists of adjusting the data power-speed in time windows of one week to a curve and then comparing the curves obtained in each week.
  - Adjustment of the data to a curve in a defined speed range is performed using penalised spline regression method
  - Compared the weekly power curves using a similar approach to functional boxplots and calculate two KPIs
    - (1) Modified Epigraph Index: Quantifies the position of current PC relative to the rest of the PCs
    - (2) Modified Band Depth : It is a measure that quantifies between how many pairs of PCs the current PC is considering all PC curve pairs
- **Change Detector Algorithm:**
  - The algorithm is intended to detect changes in statistical indicators of data series over time, such as the mean value, while the maximum number of changes to detect is an input
 
