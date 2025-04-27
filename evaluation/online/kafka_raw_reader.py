# kafka_raw_reader.py
from kafka import KafkaConsumer

def read_kafka_raw_messages(topic_name='movielog2', bootstrap_servers='localhost:9092'):
    consumer = KafkaConsumer(
        topic_name,
        bootstrap_servers=[bootstrap_servers],
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id='live-stream-reader',
        value_deserializer=lambda x: x.decode('utf-8')
    )
    print(f"✅ Connected to Kafka topic: {topic_name}")
    return consumer
