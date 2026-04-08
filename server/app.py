from social_media_env.server.app import app

if __name__ == "__main__":
    import uvicorn, os
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)
