IMAGE_NAME=audio-detect-api
PORT=8980

build:
	docker build -t $(IMAGE_NAME) .

run:
	docker run -p $(PORT):$(PORT) $(IMAGE_NAME)

dev:
	uvicorn main:app --reload --host 0.0.0.0 --port $(PORT)

push:
	docker tag $(IMAGE_NAME) your_dockerhub_username/$(IMAGE_NAME)
	docker push your_dockerhub_username/$(IMAGE_NAME)

clean:
	docker rmi $(IMAGE_NAME)

