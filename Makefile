
all:
	python -W ignore src/train.py --config config/config.yaml
	python -W ignore src/inference.py --config config/config.yaml
	cd results
	rm prediction.zip
	zip -r prediction.zip *
	cd ..

train:
	python -W ignore src/train.py --config config/config.yaml

inference:
	python -W ignore src/inference.py --config config/config.yaml
	cd results
	rm prediction.zip
	zip -r prediction.zip *
	cd ..

