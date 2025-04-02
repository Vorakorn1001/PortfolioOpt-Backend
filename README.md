## Portfolio Optimization Backend

### Setup
```bach
python -m venv venv
```

### Activate the virtual environment
```bash
# Windows
venv\Scripts\activate
# Linux / MacOS
source venv/bin/activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the server
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```