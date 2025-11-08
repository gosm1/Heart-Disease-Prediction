from pyngrok import ngrok 

port = 8501 

public_url = ngrok.connect(port).public_url