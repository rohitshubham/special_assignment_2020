import paho.mqtt.client as mqtt

mqtt_topic = "topic/request"
mqtt_host = "cloud"
mqtt_port = 1883


# This is the Publisher
def send_mqtt_requests(data):
    client = mqtt.Client()
    client.connect("cloud", mqtt_port, 60)
    client.publish("topic/request", str(data))
    client.disconnect()


send_mqtt_requests("Hello World")
