# Refresh path and start Streamlit app
$pyPath = "C:\Users\sampath\AppData\Local\Programs\Python\Python312"
$scriptsPath = "C:\Users\sampath\AppData\Local\Programs\Python\Python312\Scripts"
$env:Path = "$scriptsPath;$pyPath;$env:Path"

Write-Host "Launching Environment Classifier and Demonstrator..." -ForegroundColor Cyan
streamlit run app.py

