from airflow import DAG
    
from airflow.contrib.operators.emr_add_steps_operator import EmrAddStepsOperator
from airflow.contrib.operators.emr_create_job_flow_operator import EmrCreateJobFlowOperator
from airflow.contrib.operators.emr_terminate_job_flow_operator import EmrTerminateJobFlowOperator
from airflow.contrib.sensors.emr_step_sensor import EmrStepSensor


    
from airflow.utils.dates import days_ago
from datetime import timedelta
import os
    
DAG_ID = os.path.basename(__file__).replace(".py", "")
    
DEFAULT_ARGS = {
    'owner': 'tlc-airflow',
    'depends_on_past': False,
    'email': ['sjaqj88@gmail.com'],
    'email_on_failure': False,
    'email_on_retry': False,
}
    
SPARK_STEPS = [
    {
        'Name': 'tune parameter',
        'ActionOnFailure': 'CONTINUE',
        'HadoopJarStep': {
            'Jar': 'command-runner.jar',
            'Args': [
                'spark-submit',
                '--deploy-mode',
                'cluster',
                '--master',
                'yarn',
                '--conf',
                'spark.yarn.submit.waitAppCompletion=true',
                's3://onedayproject/scripts/tlc_tune_hyperparameter.py'
            ],
        },
    },
    {
        'Name': 'model train',
        'ActionOnFailure': 'CONTINUE',
        'HadoopJarStep': {
            'Jar': 'command-runner.jar',
            'Args': [
                'spark-submit',
                '--deploy-mode',
                'cluster',
                '--master',
                'yarn',
                '--conf',
                'spark.yarn.submit.waitAppCompletion=true',
                's3://onedayproject/scripts/tlc_train_model.py'
            ],
        },
    }
]
    
JOB_FLOW_OVERRIDES = {
    'Name': 'my-tlc-emr-cluster',
    'ReleaseLabel': 'emr-5.34.0',
    'Applications': [
        {
            'Name': 'Spark'
        },
    ],    
    'Instances': {
        'InstanceGroups': [
            {
                'Name': "Master nodes",
                'Market': 'SPOT',
                'InstanceRole': 'MASTER',
                'InstanceType': 'r5d.xlarge',
                'InstanceCount': 1,
            },
            {
                'Name': "Slave nodes",
                'Market': 'SPOT',
                'InstanceRole': 'CORE',
                'InstanceType': 'r5d.xlarge',
                'InstanceCount': 1,
            },
            # {
            #     'Name': "Slave nodes2",
            #     'Market': 'SPOT',
            #     'InstanceRole': 'TASK',
            #     'InstanceType': 'r5d.xlarge',
            #     'InstanceCount': 2,
            # }
        ],
        'KeepJobFlowAliveWhenNoSteps': False,
        'TerminationProtected': False
        # 'Ec2KeyName': 'mykeypair',
    },
    'VisibleToAllUsers': True,
    'JobFlowRole': 'EMR_EC2_DefaultRole',
    'ServiceRole': 'EMR_DefaultRole',
    
    'Tags': [
        {
            'Key': 'Name',
            'Value': 'tlc-airflow'
        }
    ]
}
    
with DAG(
    dag_id=DAG_ID,
    default_args=DEFAULT_ARGS,
    dagrun_timeout=timedelta(hours=2),
    start_date=days_ago(1),
    schedule_interval='@once',
    tags=['emr'],
) as dag:
    
    # The EmrCreateJobFlowOperator creates a cluster and 
    # stores the EMR cluster id(unique identifier) in xcom, 
    # which is a key value store used to access variables across Airflow tasks.
    cluster_creator = EmrCreateJobFlowOperator(
        task_id='create_emr_cluster', 
        job_flow_overrides=JOB_FLOW_OVERRIDES
    )
    
    # Add my steps to the EMR cluster
    step_adder = EmrAddStepsOperator(
        task_id='add_steps',
        job_flow_id="{{ task_instance.xcom_pull(task_ids='create_emr_cluster', key='return_value') }}",
        aws_conn_id='aws_default',
        steps=SPARK_STEPS,
    )
    
    # Wait for the steps to complete
    step_checker = EmrStepSensor(
        task_id='watch_step',
        job_flow_id="{{ task_instance.xcom_pull('create_emr_cluster', key='return_value') }}",
        step_id="{{ task_instance.xcom_pull(task_ids='add_steps', key='return_value')[0] }}",
        aws_conn_id='aws_default',
    )
    
    # Terminate the EMR cluster
    terminate_emr_cluster = EmrTerminateJobFlowOperator(
        task_id="terminate_emr_cluster",
        job_flow_id="{{ task_instance.xcom_pull(task_ids='create_emr_cluster', key='return_value') }}",
        aws_conn_id="aws_default",
    )
    
    
    
    cluster_creator >> step_adder >> step_checker >> terminate_emr_cluster