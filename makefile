VENV := venv

.PHONY: setup install_lib clean fclean

all: setup install_lib

setup:
	python3 -m venv $(VENV)
	@echo "Virtual environment $(VENV) created."

install_lib:
	$(VENV)/bin/pip install -r requirements.txt
	@echo "Installed lib."

clean:
	rm -rf __pycache__
	@echo "Clean ok."

fclean: clean
	rm -rf $(VENV) dist
	rm -f models/*
	rm -rf trainingData
	rm -rf temp/*
	@echo "Fclean ok."

re: fclean all
	@echo "Re ok."