from gradio_client import Client, handle_file
from PIL import Image

client = Client("http://localhost:7860/")
result = client.predict(
		image_input=handle_file(''),
		box_threshold=0.05,
		iou_threshold=0.1,
		use_paddleocr=False,
		imgsz=640,
		describe_icons=False,
		api_name="/gradio_process"
)
print(result)