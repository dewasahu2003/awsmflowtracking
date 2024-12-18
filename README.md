# MLflow Tracking Server Setup

This repository contains instructions for setting up an MLflow tracking server on AWS EC2 with S3 bucket storage, while running model training locally.

## Architecture Overview

- MLflow Tracking Server: Hosted on AWS EC2
- Artifact Storage: AWS S3 Bucket
- Model Training: Local machine
- Connection: Remote tracking via MLflow Tracking URI

## Local Setup

1. Create and activate virtual environment:
```bash
python -m venv venv
venv/Scripts/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## EC2 Server Setup

1. Connect to your EC2 instance and install required packages:
```bash
sudo apt update
sudo apt install python3-pip
sudo apt install pipenv
sudo apt install virtualenv
```

2. Create and setup MLflow directory:
```bash
mkdir mlflow
cd mlflow
```

3. Install required Python packages:
```bash
pipenv install mlflow
pipenv install awscli
pipenv install boto3
```

4. Activate virtual environment:
```bash
pipenv shell
```

5. Configure AWS credentials:
```bash
aws configure
# Follow prompts to enter:
# - AWS Access Key ID
# - AWS Secret Access Key
# - Default region
# - Default output format
```

6. Start MLflow server:
```bash
mlflow server -h 0.0.0.0 --default-artifact-root s3://mlflowtrackingbuck
```

## Environment Configuration

Set the MLflow tracking URI to point to your EC2 instance:
```bash
export MLFLOW_TRACKING_URI=http://ec2-13-127-189-46.ap-south-1.compute.amazonaws.com:5000/#
```

## Important Notes

- Replace `mlflowtrackingbuck` with your actual S3 bucket name
- Update the EC2 instance URL in the tracking URI with your actual EC2 public DNS
- Ensure proper security group settings on EC2 to allow incoming traffic on port 5000
- Make sure your AWS credentials have appropriate permissions for S3 access

## Prerequisites

- AWS Account
- EC2 instance
- S3 bucket
- Python 3.x
- pip
- AWS CLI credentials

## Security Considerations

- Keep your AWS credentials secure
- Use appropriate IAM roles and permissions
- Configure EC2 security groups properly
- Consider using environment variables for sensitive information
