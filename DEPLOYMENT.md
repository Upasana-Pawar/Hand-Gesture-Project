# Streamlit Deployment Guide

## Local Setup

### 1. Install Dependencies
```bash
pip install -r Computer-Vision-Hand-Hygiene-Scanner/requirements.txt
pip install streamlit
```

### 2. Run Locally
```bash
streamlit run app.py
```

This will open the app in your browser at `http://localhost:8501`

---

## Deploy to Streamlit Cloud (Free & Easy)

### Step 1: Push to GitHub
Make sure your code is pushed to GitHub (you already did this ✓)

### Step 2: Go to Streamlit Cloud
1. Visit https://streamlit.io/cloud
2. Click "New app"
3. Sign in with GitHub
4. Select your repository: `Hand-Gesture-Project`
5. Set Main file path: `app.py`
6. Click "Deploy"

**That's it!** Your app will be live in minutes.

---

## Deploy to Heroku

### Step 1: Install Heroku CLI
Download from https://devcenter.heroku.com/articles/heroku-cli

### Step 2: Create a Procfile
```
web: streamlit run --server.port $PORT app.py
```

### Step 3: Create setup.sh
```bash
mkdir -p ~/.streamlit/
echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml
```

### Step 4: Deploy
```bash
heroku login
heroku create your-app-name
git push heroku main
```

---

## Deploy to AWS

### Using AWS Elastic Beanstalk:
1. Install AWS CLI
2. Create `.ebextensions/python.config`:
```yaml
option_settings:
  aws:elasticbeanstalk:application:environment:
    PYTHONPATH: /var/app/current:$PYTHONPATH
commands:
  01_install_streamlit:
    command: "pip install streamlit"
```
3. Deploy with: `eb create` and `eb deploy`

---

## Deploy to Google Cloud Run

### Step 1: Create Dockerfile
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install -r Computer-Vision-Hand-Hygiene-Scanner/requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Step 2: Deploy
```bash
gcloud run deploy hand-gesture --source .
```

---

## Environment Variables

Create a `.env` file for sensitive data:
```
API_KEY=your_key_here
DEBUG=false
```

---

## Troubleshooting

### WebCam not working on cloud deployment?
- Webcam only works locally. Cloud versions support image upload.
- For production, use a video upload feature or RTMP stream.

### Model not loading?
- Ensure model files are in the correct path
- Check permissions on model files

### Memory issues?
- Streamlit Cloud has 1GB limit
- Optimize model size or use smaller images
