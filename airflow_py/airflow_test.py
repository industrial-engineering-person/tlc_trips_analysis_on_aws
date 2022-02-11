from datetime import datetime
import json

from airflow import DAG
from airflow.providers.sqlite.operators.sqlite import SqliteOperator
from airflow.providers.http.sensors.http import HttpSensor
from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from pandas import json_normalize

# OpenSea 사이트에서 NFT데이터 추출해 테이블에 저장하기 coding

default_args = {
  'start_date': datetime(2021, 1, 1),
}

# ti(task instance), xcom = cross communication
# task_ids는 여러곳에서 가져올수 있어서 배열로 작성
def _processing_nft(ti):
  assets = ti.xcom_pull(task_ids=['extract_nft']) 
  if not len(assets):
    raise ValueError("assets is empty")
  nft = assets[0]['assets'][0]
  
  # 최종 결과를 csv로 저장하기위해 pandas사용 
  # json_normalize = json을 pandas로 바꿔줌
  processed_nft = json_normalize({
    'token_id': nft['token_id'],
    'name': nft['name'],
    'image_url': nft['image_url'],
  })
  processed_nft.to_csv('/tmp/processed_nft.csv', index=None, header=False)


with DAG(dag_id='nft-pipeline',
         schedule_interval='@daily',
         default_args=default_args,
         tags=['nft'],
         catchup=False) as dag:
  
  # sqlite table 생성
  # sqliteoperator 인스턴스화 / ui상에서 확인가능한 task_id 지정
  # sqlite_conn_id 는 UI상에서 Admin탭/Connections에서 생성가능
  # 생성시 Connectionn Type에 관련 package가 없으면 따로 다운 받아야됨(pip)
  creating_table = SqliteOperator(
    task_id='creating_table',
    sqlite_conn_id='db_sqlite',
    sql='''
      CREATE TABLE IF NOT EXISTS nfts (
        token_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        image_url TEXT NOT NULL
      )
    '''
  )

  # 외부 api가 존재하는지 확인하는 센서
  is_api_available = HttpSensor(
    task_id='is_api_available',
    http_conn_id='opensea_api',
    endpoint='api/v1/assets?collection=doodles-official&limit=1'
  )
  
  # http에서 직접 데이터를 가져와서 추출하는 operator
  # json.loads(res.text) = json 문자열을 python 객체로 변환
  # 디버깅하기 편하도록 log_response 설정
  extract_nft = SimpleHttpOperator(
    task_id='extract_nft',
    http_conn_id='opensea_api',
    endpoint='api/v1/assets?collection=doodles-official&limit=1',
    method='GET',
    response_filter=lambda res: json.loads(res.text),
    log_response=True
  )
  
  # 가져온 데이터를 가공하기 위해 사용 
  process_nft = PythonOperator(
    task_id='process_nft',
    python_callable=_processing_nft
  )

  # sqlite에 저장하기위해 사용
  store_nft = BashOperator(
    task_id='store_nft',
    bash_command='echo -e ".separator ","\n.import /tmp/processed_nft.csv nfts" | sqlite3 /Users/keon/airflow/airflow.db'
  )
  
# sqlite3 airflow.db
# .tables
# select * from users;
# sqlite> .schema users
# CREATE TABLE users (
#         firstname TEXT NOT NULL,
#         lastname TEXT NOT NULL,
#         country TEXT NOT NULL,
#         username TEXT NOT NULL,
#         password TEXT NOT NULL,
#         email TEXT NOT NULL PRIMARY KEY
#       );




### dependency설정
  creating_table >> is_api_available >> extract_nft >> process_nft >> store_nft