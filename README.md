# CMT1A-SensorDataProcessing
This script provides a complete data processing pipeline for sensor data related to CMT1A, preparing it for machine learning model training with TensorFlow. The workflow covers data loading, cleaning, normalization, saving in TFRecord format, and preparing the data for model training.
# **Purpose of the Pipeline**
The purpose of this pipeline is to handle sensor data efficiently and transform it into a clean, consistent format that can be directly used for training machine learning models. Since CMT1A data often consists of readings from multiple sensors across various body parts (e.g., chest, left limb, and left thigh), it is crucial to preprocess the data accurately. This ensures that models trained on this data will perform reliably and be free from biases introduced by noise, outliers, or inconsistencies.
# Key Steps and Functions
**1. Data Loading and Cleaning**

The process has began with loading raw sensor data for multiple participants. This data typically consists of acceleration readings across different axes (X, Y, Z) for different body parts. To ensure consistency, the script uses **Pandas** to read the data from various file formats, handle missing values, and remove duplicates. This initial cleaning step lays the foundation for more advanced preprocessing.

**2. Outlier Detection and Removal**

One of the essential steps in data preprocessing is the identification and removal of outliers. Outliers are extreme values that can distort the analysis or negatively impact model training. In this pipeline, **Z-score analysis** has been applied to detect these anomalies. Any value that deviates more than 3 standard deviations from the mean is considered an outlier and is removed from the dataset. By removing these extreme values, the pipeline ensures that the data used for model training is representative of normal behavior, reducing noise and improving the reliability of downstream models.

**3. Data Normalization**

Once the data is cleaned, the pipeline proceeds to normalize the features. Normalization is critical because it ensures that all features (sensor readings from different axes and body parts) are on the same scale. I have used **MinMaxScaler** from **scikit-learn** to transform all acceleration values into a consistent [0, 1] range. This standardization makes the data suitable for machine learning algorithms that are sensitive to feature scaling, enhancing model performance and training efficiency.

**4. Saving Data as TFRecord Files**

One of the main goals of the pipeline is to transform the processed data into TensorFlow's preferred format: **TFRecord**. This format is efficient for storing large datasets and is optimized for TensorFlow training. Each row of the normalized data is converted into a serialized TFRecord example using TensorFlow’s **tf.train.Feature** and **tf.train.Example** functionalities. Depending on the dataset size, the pipeline offers options to write the data in batches or in parallel to speed up the saving process. The resulting .tfrecord files are easy to manage and can be seamlessly integrated into TensorFlow workflows.

**5. Verification of Data Integrity**

To ensure that the data transformation has not altered the original information, the pipeline includes a verification step. It compares the data loaded from TFRecord files with the original normalized dataset to confirm that they match. This comparison is done using visualizations of data distributions, which verify that the saved and loaded data are identical. This step is crucial to confirm that no information is lost during the process of saving to and reading from TFRecord files.
# Required Packages
1. TensorFlow
2. Pandas
3. NumPy
4. SciPy
5. Matplotlib
6. Scikit-learn
7. Python 3.12.6 with CUDA 11.2
# **Contributors**
Contributions are welcome! Feel free to open issues or submit pull requests.
# Licence
This project is licensed under the Apache 2.0 License - see the LICENSE file for details.
