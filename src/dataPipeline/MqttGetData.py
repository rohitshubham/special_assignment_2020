import paho.mqtt.client as mqtt

mqtt_topic = "topic/request"
mqtt_host = "localhost"
mqtt_port = 1883

client = mqtt.Client()
client.connect(mqtt_host, mqtt_port)


def on_connect(client, userdata, flags, rc):
    print(f"Connected to {mqtt_host} with result code {str(rc)}.")
    client.subscribe(mqtt_topic)
    print(f"Subscribed to topic {mqtt_topic} on {mqtt_host}")


def on_message(client, userdata, msg):
    try:
        print(str(msg.payload.decode()))
    except Exception as e:
        print(f"some error : {e}")


client.on_connect = on_connect
client.on_message = on_message

client.loop_forever()
