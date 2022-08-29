# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either the provided dog breed classication data set or one of your choice.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom.
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project.

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 

## Hyperparameter Tuning
This project employs a deep learning model with Pytorch framework, as deep learning tends to performs well in classification tasks with a large amount of data. 

Learning rate and batch size are selected for HP tuning as shown below. Learning rate is important in model tuning as a small learning rate can be time-consuming while a overly-large figure can cause overshoots and lead to non-optimal resutls. Batch size is also important as it determines the error gradient of training process. 

```python
hyperparameter_ranges = {
    "lr": ContinuousParameter(0.001, 0.1),
    "batch-size": CategoricalParameter([32, 64, 128, 256]),
}
```

#### Completed training jobs
[]()

#### Training metrics
```python
{
    "additional_framework_parameters": {},
    "channel_input_dirs": {
        "train": "/opt/ml/input/data/train"
    },
    "current_host": "algo-1",
    "framework_module": "sagemaker_pytorch_container.training:main",
    "hosts": [
        "algo-1"
    ],
    "hyperparameters": {
        "batch-size": 256,
        "lr": "0.01390030725982051"
    },
    "input_config_dir": "/opt/ml/input/config",
    "input_data_config": {
        "train": {
            "TrainingInputMode": "File",
            "S3DistributionType": "FullyReplicated",
            "RecordWrapperType": "None"
        }
    },
    "input_dir": "/opt/ml/input",
    "is_master": true,
    "job_name": "pytorch-training-2022-08-29-00-18-40-802",
    "log_level": 20,
    "master_hostname": "algo-1",
    "model_dir": "/opt/ml/model",
    "module_dir": "s3://sagemaker-us-east-1-705959383387/pytorch-training-2022-08-29-00-18-40-802/source/sourcedir.tar.gz",
    "module_name": "train_model1",
    "network_interface_name": "eth0",
    "num_cpus": 4,
    "num_gpus": 1,
    "output_data_dir": "/opt/ml/output/data",
    "output_dir": "/opt/ml/output",
    "output_intermediate_dir": "/opt/ml/output/intermediate",
    "resource_config": {
        "current_host": "algo-1",
        "current_instance_type": "ml.g4dn.xlarge",
        "current_group_name": "homogeneousCluster",
        "hosts": [
            "algo-1"
        ],
        "instance_groups": [
            {
                "instance_group_name": "homogeneousCluster",
                "instance_type": "ml.g4dn.xlarge",
                "hosts": [
                    "algo-1"
                ]
            }
        ],
        "network_interface_name": "eth0"
    },
    "user_entry_point": "train_model.py"
}
```

#### Best Hyperparameters
```python
{
 'batch-size': '"256"',
 'lr': '0.01390030725982051',
}
```


Remember that your README should:
- Include a screenshot of completed training jobs
- Logs metrics during the training process
- Tune at least two hyperparameters
- Retrieve the best best hyperparameters from all your training jobs

## Debugging and Profiling
During the model training, AWS Sagemaker debugger and profiler were employed for debugging and profilling purposes. They were configured as follows:
```python
rules = [
    ProfilerRule.sagemaker(rule_configs.ProfilerReport()),
    Rule.sagemaker(rule_configs.loss_not_decreasing()),
    Rule.sagemaker(rule_configs.vanishing_gradient()),
    Rule.sagemaker(rule_configs.overfit()),
    Rule.sagemaker(rule_configs.overtraining()),
    Rule.sagemaker(rule_configs.poor_weight_initialization()),
]

profiler_config = ProfilerConfig(
    system_monitor_interval_millis=500, 
    framework_profile_params=FrameworkProfile(num_steps=10)
)

debugger_config = DebuggerHookConfig(
    hook_parameters={"train.save_interval": "100", "eval.save_interval": "10"}
)
```

### Results
**TODO**: What are the results/insights did you get by profiling/debugging your model?
From the profiler report, we can see that most of operator usage was spent on IO operations like "copy_" and "to" for both GPU and CPU. This makes sense as given that large amount of data were going through during the training process. 

It's also noteworthy that GPUMemoryIncrease rule was triggered the most during our training, and thus we should consider choosing a larger instance type with more memory if footprint is close to maximum available memory. 

**TODO** Remember to provide the profiler html/pdf file in your submission.


## Model Deployment
The inference model is a Pytorch neural network model for dog breed classification tasks. Users can submit an inference request by passing an serialized image data of a size of 224. This can be done as the example below:
```python
import torchvision.transforms as transform
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

image = Image.open(img_path)
img_tensor = transform(image).unsqueeze(0) 

```

**TODO** Remember to provide a screenshot of the deployed active endpoint in Sagemaker.

## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.
