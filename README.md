MLFLOW TRACKING SERVER WILL HAPPEN ON AWS EC2
AND ALL PROCESSING WILL HAPPEN ON THIS LAPTOP


`running aws ec2 instance`

sudo apt update
sudo apt install python3-pip
sudo apt install pipenv
sudo apt install virtualenv

mkdir mlflow
cd mlflow
pipenv install mlflow
pipenv install awscli
pipenv install boto3
pipenv install shell
pipenv shell

#settiing aws cred

aws configure

mlflow server -h 0.0.0.0 --default-artifact-root s3://mlflowtrackingbuck

#setting env variable
export MLFLOW_TRACKING_URI=http://ec2-13-127-189-46.ap-south-1.compute.amazonaws.com:5000/#   a w s m f l o w t r a c k i n g  
 