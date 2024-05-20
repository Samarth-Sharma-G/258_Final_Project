import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer
import mlflow
import mlflow.tensorflow
import os
import tensorflow as tf
from mlflow.models.signature import infer_signature
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd 
from azureml.core import Workspace, Dataset, Datastore
from azureml.exceptions import UserErrorException    
from mlflow.deployments import get_deploy_client
import json
from sklearn.metrics import accuracy_score
from azure.core.exceptions import ResourceNotFoundError

class CNNTrainer2:
    def __init__(self, pathX, pathY):
        self.Xpath = pathX
        self.Ypath = pathY
        self.X = None
        self.Y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.champion_model = None
        self.champion_model_name = None
        self.tracking_tracking_uri = "azureml://westus2.api.azureml.ms/mlflow/v1.0/subscriptions/7daeeb08-64ac-4094-abc8-02d2a530cc6e/resourceGroups/CMPE258/providers/Microsoft.MachineLearningServices/workspaces/Team-pi"
        # Experiment name will be followed by date time
        self.initialization_time = str(pd.Timestamp.now())
        self.experiment_name = 'waffer_defect_' + self.initialization_time
        mlflow.set_experiment(self.experiment_name)
        # Load the data and perform the train test split
        self.load_data_and_perform_split()
        mlflow.set_tracking_uri(self.tracking_tracking_uri)
        self.register_model_name = 'waffer-defect'

    
    def load_preprocessed_data(self):
        self.X = pd.read_pickle(self.Xpath)
        self.Y = pd.read_pickle(self.Ypath)

    def perform_train_test_split(self, split=0.2, seed=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=split, random_state=seed)

    def model1(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(26, 26, 1)),
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def model2(self):
        model = tf.keras.models.Sequential([
                tf.keras.layers.Input(shape=(26, 26, 1)),
                tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
                ])
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def model3(self):
        model = tf.keras.models.Sequential([
                tf.keras.layers.Input(shape=(26, 26, 1)),
                tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
                ])

        model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_model(self, model, model_name, epochs=1):
        with mlflow.start_run(run_name=model_name):
            early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            history = model.fit(self.X_train, self.y_train, epochs=epochs, validation_split=0.1, callbacks=[early_stopping])
            loss, accuracy = model.evaluate(self.X_test, self.y_test)
            print(f'Test accuracy: {accuracy:.4f}')
            
            # Log parameters and metrics
            mlflow.log_param("epochs", epochs)
            mlflow.log_metric("loss", loss)
            mlflow.log_metric("accuracy", accuracy)

            # Log the model
            input_example = self.X_train[:1]
            output_example = model.predict(input_example)
            signature = infer_signature(input_example, output_example)

            mlflow.tensorflow.log_model(model, artifact_path=model_name, signature=signature)
            # Enable autologging
            mlflow.tensorflow.autolog()
            
        return model, history

    def get_champion_model(self):
        # Example logic to select the champion model based on accuracy
        accuracies = {}
        trained_models = {}
        models = [self.model1, self.model2, self.model3]
        for i, model_fn in enumerate(models, start=1):
            model = model_fn()
            model_name = f'model{i}'
            trained_model, history = self.train_model(model, model_name)
            accuracies[f'model{i}'] = max(history.history['val_accuracy'])
            trained_models[f'model{i}'] = trained_model
        self.champion_model_name = max(accuracies, key=accuracies.get)
        self.champion_model = trained_models[self.champion_model_name]
        print(f'Champion Model: {self.champion_model_name}')
        return self.champion_model

    def register_champion_model(self):
        # Placeholder for actual registration logic
        run_id = mlflow.search_runs(experiment_ids=mlflow.get_experiment_by_name(self.experiment_name).experiment_id)
        run_id = run_id[run_id['tags.mlflow.runName'] == self.champion_model_name]['run_id'].values[0]
        mlflow.register_model(f"runs:/{run_id}/{self.champion_model_name}", self.register_model_name)
    
    def load_data_and_perform_split(self):
        self.load_preprocessed_data()
        self.perform_train_test_split()
    
    def train_register_champion_model(self):
        self.get_champion_model()
        self.register_champion_model()

#loading the data from the blob storage
def get_file_from_blob(filename):
    try:
        dataset = Dataset.File.from_files(path=(datastore, filename))
        dataset.download(target_path='.', overwrite=False)
    except UserErrorException as e:
        print('File Already Exists in the directory')

def check_if_endpoint_exists(endpoint_name):
    try:
        #checking if the endpoint exists
        deployment_client.get_endpoint(endpoint_name)
        return True
    # except mlflow.exceptions.MlflowException as e:
    #     print(f"Error: {e}")
    #     return False
    except ResourceNotFoundError:
        return False

def create_endpoint(endpoint_name,endpoint_config):
    #writing the config to a json file so that we can pass it to the create_deployment function
    endpoint_config_path = "endpoint_config.json"
    with open(endpoint_config_path, "w") as outfile:
        outfile.write(json.dumps(endpoint_config))
    endpoint = deployment_client.create_endpoint(
        name=endpoint_name,
        config={"endpoint-config-file": endpoint_config_path},
    )

def create_deployment(deployment_name, endpoint_name ,deploy_config,model_name,version):
    deployment_config_path = "deployment_config.json"
    with open(deployment_config_path, "w") as outfile:
        outfile.write(json.dumps(deploy_config))
    blue_deployment = deployment_client.create_deployment(
        name=deployment_name,
        endpoint=endpoint_name,
        model_uri=f"models:/{model_name}/{version}",
        config={"deploy-config-file": deployment_config_path},
    )

def allocate_traffic_to_deployment(traffic_config, endpoint_name,deployment_name):
    traffic_config_path = "traffic_config.json"
    with open(traffic_config_path, "w") as outfile:
        outfile.write(json.dumps(traffic_config))
    #updatin
    deployment_client.update_endpoint(
        endpoint=endpoint_name,
        config={"endpoint-config-file": traffic_config_path},
    )

def get_prod_model_name(endpoint_name):
    ep = deployment_client.get_endpoint(endpoint_name)
    d_name = list(ep['properties']['traffic'].keys())[0]
    print(d_name)
    dep = deployment_client.get_deployment(d_name,endpoint=endpoint_name)
    model_info = dep['properties']['environmentVariables']['AZUREML_MODEL_DIR'].split('/')
    model_version = model_info[-1]# after splitting the last element is the model version
    model_name = model_info[-2]# after splitting the second last element is the model name
    print(f"Model Name: {model_name} Model Version: {model_version}")
    return model_name,model_version

def load_model_from_registry(model_name, model_version):
    # model = client.get_model_version(model_name, model_version)
    path = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(path)
    return model


def load_data():
    X = pd.read_pickle(images_extracted_file)
    Y = pd.read_pickle(images_labels_file)
    return train_test_split(X, Y, test_size=0.2, random_state=42)

def get_latest_version(model_name):
    runs = client.search_model_versions(f"name='{model_name}'")
    return max([r.version for r in runs])    

def comapare_performance_perforamce(model1,model2):
    X_train, X_test, y_train, y_test = load_data()
    y1 = model1.predict(X_test)
    y2 = model2.predict(X_test)
    #models give out prodability that we need to convert to 0 or 1
    y1 = [1 if i > 0.5 else 0 for i in y1]
    y2 = [1 if i > 0.5 else 0 for i in y2]
    acc1 = accuracy_score(y_test, y1)
    acc2 = accuracy_score(y_test, y2)
    print(f"Model 1 Accuracy: {acc1} Model 2 Accuracy: {acc2}")
    if acc1 > acc2:
        return 1
    else:
        return 2
    
#orchestration block 
if __name__ == '__main__':
    #setting up the workspace
    subscription_id = '7daeeb08-64ac-4094-abc8-02d2a530cc6e'
    resource_group = 'CMPE258'
    workspace_name = 'Team-pi'

    workspace = Workspace(subscription_id, resource_group, workspace_name)
    datastore = Datastore.get(workspace, "workspaceblobstore")

    # # Getting the data from the blob storage
    images_extracted_file =  'images_extracted.pkl'
    images_labels_file = 'images_labels.pkl'
    get_file_from_blob(images_extracted_file)
    get_file_from_blob(images_labels_file)
    
    # # Training the model
    trainer = CNNTrainer2(images_extracted_file,images_labels_file)
    #training a number of model and then registering the champion model
    trainer.train_register_champion_model()
    
    #initializing clients needed for deployment
    client = mlflow.tracking.MlflowClient()
    deployment_client = get_deploy_client(mlflow.get_tracking_uri())
    endpoint_name = "team-pi-vtzdu"
    model_name = "waffer-defect"
    latest_version = get_latest_version(model_name)
    endpoint_config = {
        "auth_mode": "key",
        "identity": {
            "type": "system_assigned"
        }
        }
    blue_deployment_name = "waffer-defect-1"
    deploy_config = {
        "instance_type": "Standard_D2as_v4",
        "instance_count": 1,
    }
    traffic_config = {"traffic": {blue_deployment_name: 100}}

    if not check_if_endpoint_exists(endpoint_name):
        print('Endpoint does not exists Using the latest version to create one')
        create_endpoint(endpoint_name,endpoint_config)
        create_deployment(blue_deployment_name,endpoint_name,deploy_config,model_name,latest_version)
        allocate_traffic_to_deployment(traffic_config,endpoint_name,blue_deployment_name)
    else:
        # print('deployment exists')
        prod_model_name, prod_model_version = get_prod_model_name(endpoint_name)
        prod_model = load_model_from_registry(prod_model_name, prod_model_version)
        challenger_model = load_model_from_registry(prod_model_name, latest_version)
        better_model = comapare_performance_perforamce(prod_model,challenger_model)
        if better_model == 2:
            #redeply else do nothing
            print("Challenger model is better")
            new_deployment_name = 'challenger-deployment'
            create_deployment(new_deployment_name,endpoint_name,deploy_config,model_name,latest_version)
            allocate_traffic_to_deployment(traffic_config,endpoint_name,new_deployment_name)
            #delete the old deployment
            deployment_client.delete_deployment(blue_deployment_name,endpoint_name)
        else:
            print("Prod model is better")
            print("No need for updation")

    
        
    
    
