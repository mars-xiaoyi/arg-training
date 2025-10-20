# Setup Environment
- Install Anaconda
- Create and Active Conda Virtual Environment
  - ```conda create --name rag python=3.13```
  - ```conda activate rag```
  - ```pip install -r dependencies.txt```

# Use Google Cloud APIs (VertexAI)
1. Create a google cloud project (if you don't have one).
2. Enable Vertex AI API: In your project, navigate to "APIs & Services" > "Enabled APIs & Services" and enable the "Vertex AI API."
3. Enable billing on the project.
4. Make sure you have at least "Vertex API User" role on you AMI profile: Check "IAM & Admin" > "Service Accounts".
5. Install GLI in your enviornment: ```pip install gli```
6. Login your Google Could account: ```gcloud auth application-default login```