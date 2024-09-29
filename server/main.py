from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI()

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Sample steps with descriptions (easy to add/remove)
steps = [
    {"name": "Raise Your Arms", "description": "Lift both arms above your head"},
    {"name": "Spin Around", "description": "Turn in a full circle"},
    {"name": "Stand on One Leg", "description": "Balance on either leg for 3 seconds"},
    {"name": "Touch Your Toes", "description": "Bend down and touch your toes"},
    {"name": "Jump", "description": "Do a small jump"}
]
current_step = 0
game_status = "ongoing"

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    global current_step, game_status
    
    if current_step >= len(steps):
        game_status = "complete"
        step_name = "Verification Complete"
        step_description = "You have successfully completed all steps."
    else:
        step_name = steps[current_step]["name"]
        step_description = steps[current_step]["description"]
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "current_step": min(current_step, len(steps)),
        "total_steps": len(steps),
        "step_name": step_name,
        "step_description": step_description,
        "game_status": game_status
    })

@app.post("/update/{step}")
async def update_step(step: int):
    global current_step, game_status
    if 0 <= step <= len(steps):
        current_step = step
        if current_step == len(steps):
            game_status = "complete"
        else:
            game_status = "success"
    return {"status": game_status}

@app.post("/failed")
async def game_failed():
    global game_status
    game_status = "failed"
    return {"status": "failed"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
