from azureml.core import Workspace
from azureml.core.authentication import AzureCliAuthentication
import os


# Create the workspace by supplying the specified parameters
ws = Workspace.create(
    name='<WORKSPACE_NAME>',
    subscription_id='<SUBSCRIPTION_ID>',
    resource_group='<RESOURCE_GROUP_NAME>',
    location='<WORKSPACE_LOCATION>',
    create_resource_group=True,
    sku='basic',
    exist_ok=True,
    auth=AzureCliAuthentication()
)
print(ws.get_details())

# Write the details of the workspace to a configuration file in the project root
project_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../')
ws.write_config(path=project_root)
